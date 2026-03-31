; ─────────────────────────────────────────────────────────────────────────────
; CV2T Inno Setup Installer Script
;
; Produces a single CV2T-Setup-2.0.0.exe that handles:
;   - File extraction (from PyInstaller dist/cv2t/ output)
;   - Engine selection (Whisper, Canary, or Both)
;   - Model download via bundled cv2t.exe download-model
;   - Desktop + Start Menu shortcuts
;   - Data migration from previous installs
;   - Windows Defender exclusions
;   - Silent / unattended mode with /ENGINE= parameter
;
; Build:
;   pyinstaller cv2t.spec
;   iscc installer\cv2t-setup.iss
;
; Requires Inno Setup 6.x — https://jrsoftware.org/isdl.php
; ─────────────────────────────────────────────────────────────────────────────

#define MyAppName "CV2T"
#define MyAppVersion "2.0.0"
#define MyAppPublisher "kwp490"
#define MyAppURL "https://github.com/kwp490/cv2t"
#define MyAppExeName "cv2t.exe"

[Setup]
AppId={{B7E9F3A1-9C2D-4E5F-8A1B-3D7C6E4F2A9B}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
DefaultDirName={autopf}\CV2T
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
OutputDir=Output
OutputBaseFilename=CV2T-Setup-{#MyAppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayName={#MyAppName}
MinVersion=10.0
SetupLogging=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
; Bundle entire PyInstaller output directory
Source: "..\dist\cv2t\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Dirs]
; Create writable data subdirectories
Name: "{app}\models";  Permissions: users-modify
Name: "{app}\config";  Permissions: users-modify
Name: "{app}\logs";    Permissions: users-modify
Name: "{app}\temp";    Permissions: users-modify

[Icons]
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; \
    WorkingDir: "{app}"; Comment: "CV2T — Voice to Text"
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; \
    WorkingDir: "{app}"; Comment: "CV2T — Voice to Text"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"

[Run]
; Windows Defender exclusion (best-effort, silent)
Filename: "powershell.exe"; \
    Parameters: "-NoProfile -ExecutionPolicy Bypass -Command ""Add-MpPreference -ExclusionPath '{app}' -ErrorAction SilentlyContinue; Add-MpPreference -ExclusionProcess '{app}\{#MyAppExeName}' -ErrorAction SilentlyContinue"""; \
    Flags: runhidden waituntilterminated; StatusMsg: "Configuring Windows Defender exclusions..."

; Offer to launch after install
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; \
    Flags: nowait postinstall skipifsilent; WorkingDir: "{app}"

[UninstallDelete]
; Clean up logs and temp on uninstall (models/config are left for the user)
Type: filesandordirs; Name: "{app}\logs"
Type: filesandordirs; Name: "{app}\temp"

[UninstallRun]
; Remove Defender exclusions on uninstall
Filename: "powershell.exe"; \
    Parameters: "-NoProfile -ExecutionPolicy Bypass -Command ""Remove-MpPreference -ExclusionPath '{app}' -ErrorAction SilentlyContinue; Remove-MpPreference -ExclusionProcess '{app}\{#MyAppExeName}' -ErrorAction SilentlyContinue"""; \
    Flags: runhidden waituntilterminated

[Code]
// ── Engine selection state ──────────────────────────────────────────────────
var
  EnginePage: TWizardPage;
  EngineWhisperRadio: TNewRadioButton;
  EngineCanaryRadio: TNewRadioButton;
  EngineBothRadio: TNewRadioButton;
  GpuInfoLabel: TNewStaticText;
  DownloadPage: TOutputProgressWizardPage;

// ── GPU detection ───────────────────────────────────────────────────────────
function DetectGPU: String;
var
  ResultCode: Integer;
  TempFile: String;
  Lines: TArrayOfString;
begin
  Result := '';
  TempFile := ExpandConstant('{tmp}\gpu_detect.txt');
  // Run nvidia-smi and capture output to a temp file
  if Exec('cmd.exe',
      '/C nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits > "' + TempFile + '" 2>&1',
      '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
  begin
    if (ResultCode = 0) and LoadStringsFromFile(TempFile, Lines) then
    begin
      if GetArrayLength(Lines) > 0 then
        Result := Trim(Lines[0]);
    end;
  end;
  DeleteFile(TempFile);
end;

// ── Parse /ENGINE= command-line parameter ───────────────────────────────────
function GetEngineParam: String;
var
  I: Integer;
  Param: String;
begin
  Result := '';
  for I := 1 to ParamCount do
  begin
    Param := ParamStr(I);
    if (Pos('/ENGINE=', Uppercase(Param)) = 1) then
    begin
      Result := Lowercase(Copy(Param, 9, Length(Param)));
      Break;
    end;
  end;
end;

// ── Selected engine helpers ─────────────────────────────────────────────────
function InstallWhisper: Boolean;
var
  Param: String;
begin
  Param := GetEngineParam;
  if Param <> '' then
    Result := (Param = 'whisper') or (Param = 'both')
  else
    Result := EngineWhisperRadio.Checked or EngineBothRadio.Checked;
end;

function InstallCanary: Boolean;
var
  Param: String;
begin
  Param := GetEngineParam;
  if Param <> '' then
    Result := (Param = 'canary') or (Param = 'both')
  else
    Result := EngineCanaryRadio.Checked or EngineBothRadio.Checked;
end;

function DefaultEngineName: String;
begin
  if InstallCanary and (not InstallWhisper) then
    Result := 'canary'
  else
    Result := 'whisper';
end;

// ── Create Engine Selection wizard page ─────────────────────────────────────
procedure CreateEngineSelectionPage;
var
  GpuName: String;
  Lbl: TNewStaticText;
  TopPos: Integer;
begin
  EnginePage := CreateCustomPage(wpSelectDir,
    'Select Speech Engine',
    'Choose which speech recognition engine(s) to install.' + #13#10 +
    'Models will be downloaded after file installation (~1–3 GB).');

  TopPos := 0;

  // GPU info
  GpuInfoLabel := TNewStaticText.Create(EnginePage);
  GpuInfoLabel.Parent := EnginePage.Surface;
  GpuInfoLabel.Left := 0;
  GpuInfoLabel.Top := TopPos;
  GpuInfoLabel.Width := EnginePage.SurfaceWidth;
  GpuInfoLabel.AutoSize := False;
  GpuInfoLabel.WordWrap := True;

  GpuName := DetectGPU;
  if GpuName <> '' then
    GpuInfoLabel.Caption := 'GPU detected: ' + GpuName
  else
    GpuInfoLabel.Caption := 'No NVIDIA GPU detected — transcription will fall back to CPU (slower).';
  GpuInfoLabel.Height := ScaleY(30);
  TopPos := TopPos + ScaleY(40);

  // Whisper radio
  EngineWhisperRadio := TNewRadioButton.Create(EnginePage);
  EngineWhisperRadio.Parent := EnginePage.Surface;
  EngineWhisperRadio.Left := 0;
  EngineWhisperRadio.Top := TopPos;
  EngineWhisperRadio.Width := EnginePage.SurfaceWidth;
  EngineWhisperRadio.Caption := 'Whisper only  (recommended — fast, ~3 GB VRAM)';
  EngineWhisperRadio.Checked := True;
  TopPos := TopPos + ScaleY(28);

  // Whisper description
  Lbl := TNewStaticText.Create(EnginePage);
  Lbl.Parent := EnginePage.Surface;
  Lbl.Left := ScaleX(20);
  Lbl.Top := TopPos;
  Lbl.Width := EnginePage.SurfaceWidth - ScaleX(20);
  Lbl.Caption := 'CTranslate2 backend. No Python required. Works out of the box.';
  Lbl.Font.Color := clGray;
  TopPos := TopPos + ScaleY(30);

  // Canary radio
  EngineCanaryRadio := TNewRadioButton.Create(EnginePage);
  EngineCanaryRadio.Parent := EnginePage.Surface;
  EngineCanaryRadio.Left := 0;
  EngineCanaryRadio.Top := TopPos;
  EngineCanaryRadio.Width := EnginePage.SurfaceWidth;
  EngineCanaryRadio.Caption := 'Canary only  (higher accuracy, ~5 GB VRAM)';
  TopPos := TopPos + ScaleY(28);

  // Canary description
  Lbl := TNewStaticText.Create(EnginePage);
  Lbl.Parent := EnginePage.Surface;
  Lbl.Left := ScaleX(20);
  Lbl.Top := TopPos;
  Lbl.Width := EnginePage.SurfaceWidth - ScaleX(20);
  Lbl.AutoSize := False;
  Lbl.WordWrap := True;
  Lbl.Height := ScaleY(30);
  Lbl.Caption := 'NVIDIA NeMo/PyTorch backend. Requires a bundled Python environment (.venv) in the release package.';
  Lbl.Font.Color := clGray;
  TopPos := TopPos + ScaleY(40);

  // Both radio
  EngineBothRadio := TNewRadioButton.Create(EnginePage);
  EngineBothRadio.Parent := EnginePage.Surface;
  EngineBothRadio.Left := 0;
  EngineBothRadio.Top := TopPos;
  EngineBothRadio.Width := EnginePage.SurfaceWidth;
  EngineBothRadio.Caption := 'Both engines';
  TopPos := TopPos + ScaleY(28);

  // Both description
  Lbl := TNewStaticText.Create(EnginePage);
  Lbl.Parent := EnginePage.Surface;
  Lbl.Left := ScaleX(20);
  Lbl.Top := TopPos;
  Lbl.Width := EnginePage.SurfaceWidth - ScaleX(20);
  Lbl.Caption := 'Install both engines. You can switch between them in Settings.';
  Lbl.Font.Color := clGray;
end;

// ── Skip engine page in silent mode if /ENGINE= is given ────────────────────
function ShouldSkipPage(PageID: Integer): Boolean;
begin
  Result := False;
  if (PageID = EnginePage.ID) and (GetEngineParam <> '') then
    Result := True;
end;

// ── Recursive directory copy helper ─────────────────────────────────────────
procedure DirectoryCopy(SourceDir, DestDir: String);
var
  FindRec: TFindRec;
  SourcePath, DestPath: String;
begin
  if not ForceDirectories(DestDir) then
    Exit;
  if FindFirst(SourceDir + '\*', FindRec) then
  begin
    try
      repeat
        if (FindRec.Name = '.') or (FindRec.Name = '..') then
          Continue;
        SourcePath := SourceDir + '\' + FindRec.Name;
        DestPath := DestDir + '\' + FindRec.Name;
        if (FindRec.Attributes and FILE_ATTRIBUTE_DIRECTORY) <> 0 then
          DirectoryCopy(SourcePath, DestPath)
        else
          FileCopy(SourcePath, DestPath, False);
      until not FindNext(FindRec);
    finally
      FindClose(FindRec);
    end;
  end;
end;

// ── Migrate data from previous install locations ────────────────────────────
procedure MigrateOldData;
var
  OldSettings, NewSettings: String;
  OldModelsDir, NewEngineDir: String;
  FindRec: TFindRec;
  OldLogDir, OldLog, NewLog: String;
  LogFiles: array[0..2] of String;
  I: Integer;
begin
  // Migrate settings.json from %APPDATA%\CV2T
  OldSettings := ExpandConstant('{userappdata}\CV2T\settings.json');
  NewSettings := ExpandConstant('{app}\config\settings.json');
  if FileExists(OldSettings) and (not FileExists(NewSettings)) then
    FileCopy(OldSettings, NewSettings, False);

  // Migrate models from %LOCALAPPDATA%\CV2T\models
  OldModelsDir := ExpandConstant('{localappdata}\CV2T\models');
  if DirExists(OldModelsDir) then
  begin
    if FindFirst(OldModelsDir + '\*', FindRec) then
    begin
      try
        repeat
          if (FindRec.Attributes and FILE_ATTRIBUTE_DIRECTORY) <> 0 then
          begin
            if (FindRec.Name <> '.') and (FindRec.Name <> '..') then
            begin
              NewEngineDir := ExpandConstant('{app}\models\') + FindRec.Name;
              if not DirExists(NewEngineDir) then
                DirectoryCopy(OldModelsDir + '\' + FindRec.Name, NewEngineDir);
            end;
          end;
        until not FindNext(FindRec);
      finally
        FindClose(FindRec);
      end;
    end;
  end;

  // Migrate log files
  OldLogDir := ExpandConstant('{userappdata}\CV2T');
  LogFiles[0] := 'cv2t.log';
  LogFiles[1] := 'cv2t.log.1';
  LogFiles[2] := 'cv2t.log.2';
  for I := 0 to 2 do
  begin
    OldLog := OldLogDir + '\' + LogFiles[I];
    NewLog := ExpandConstant('{app}\logs\') + LogFiles[I];
    if FileExists(OldLog) and (not FileExists(NewLog)) then
      FileCopy(OldLog, NewLog, False);
  end;
end;

// ── Write settings.json with selected engine ────────────────────────────────
procedure WriteDefaultSettings;
var
  SettingsFile: String;
  Json: String;
begin
  SettingsFile := ExpandConstant('{app}\config\settings.json');
  // Only write if settings.json doesn't already exist (preserve user config)
  if not FileExists(SettingsFile) then
  begin
    Json := '{' + #13#10 +
            '  "engine": "' + DefaultEngineName + '"' + #13#10 +
            '}';
    SaveStringToFile(SettingsFile, Json, False);
  end;
end;

// ── Download models via cv2t.exe ────────────────────────────────────────────
procedure DownloadModels;
var
  ExePath, ModelsDir: String;
  ResultCode: Integer;
begin
  ExePath := ExpandConstant('{app}\{#MyAppExeName}');
  ModelsDir := ExpandConstant('{app}\models');

  DownloadPage := CreateOutputProgressPage('Downloading Models',
    'Please wait while speech recognition models are downloaded...');

  if InstallWhisper then
  begin
    DownloadPage.SetText('Downloading Whisper model (large-v3-turbo)...', '');
    DownloadPage.SetProgress(0, 100);
    DownloadPage.Show;
    try
      Exec(ExePath, 'download-model --engine whisper --target-dir "' + ModelsDir + '"',
           '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
      if ResultCode <> 0 then
        MsgBox('Whisper model download failed (exit code ' + IntToStr(ResultCode) + ').' + #13#10 +
               'You can download it later by running:' + #13#10 +
               '"' + ExePath + '" download-model --engine whisper',
               mbError, MB_OK);
    except
      MsgBox('Could not start model download. You can download models later from the application.',
             mbError, MB_OK);
    end;
    DownloadPage.SetProgress(50, 100);
  end;

  if InstallCanary then
  begin
    DownloadPage.SetText('Downloading Canary model (nvidia/canary-qwen-2.5b)...', '');
    if not InstallWhisper then
    begin
      DownloadPage.SetProgress(0, 100);
      DownloadPage.Show;
    end;
    try
      Exec(ExePath, 'download-model --engine canary --target-dir "' + ModelsDir + '"',
           '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
      if ResultCode <> 0 then
        MsgBox('Canary model download failed (exit code ' + IntToStr(ResultCode) + ').' + #13#10 +
               'You can download it later by running:' + #13#10 +
               '"' + ExePath + '" download-model --engine canary',
               mbError, MB_OK);
    except
      MsgBox('Could not start model download. You can download models later from the application.',
             mbError, MB_OK);
    end;
  end;

  DownloadPage.SetProgress(100, 100);
  DownloadPage.Hide;
end;

// ── Post-install orchestration ──────────────────────────────────────────────
procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    MigrateOldData;
    WriteDefaultSettings;
    DownloadModels;
  end;
end;

// ── InitializeWizard: create custom pages ───────────────────────────────────
procedure InitializeWizard;
begin
  CreateEngineSelectionPage;
end;
