; ─────────────────────────────────────────────────────────────────────────────
; CV2T Inno Setup Installer Script
;
; Produces a single CV2T-Setup-3.0.0.exe that handles:
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
#define MyAppVersion "3.0.0"
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
; Always clean up logs, temp, and config on uninstall.
; Models are handled by CurUninstallStepChanged (user is prompted).
Type: filesandordirs; Name: "{app}\logs"
Type: filesandordirs; Name: "{app}\temp"
Type: filesandordirs; Name: "{app}\config"

[UninstallRun]
; Remove Defender exclusions on uninstall
Filename: "powershell.exe"; \
    Parameters: "-NoProfile -ExecutionPolicy Bypass -Command ""Remove-MpPreference -ExclusionPath '{app}' -ErrorAction SilentlyContinue; Remove-MpPreference -ExclusionProcess '{app}\{#MyAppExeName}' -ErrorAction SilentlyContinue"""; \
    Flags: runhidden waituntilterminated; RunOnceId: "DefenderExclusions"

[Code]
// ── Engine selection state ──────────────────────────────────────────────────
var
  EnginePage: TWizardPage;
  GpuInfoLabel: TNewStaticText;
  DownloadPage: TOutputProgressWizardPage;
  SummaryPage: TWizardPage;
  SummaryMemo: TNewMemo;
  DetectedGPU: String;

// ── GPU detection ───────────────────────────────────────────────────────────
function DetectGPU: String;
var
  ResultCode: Integer;
  TempFile: String;
  Lines: TArrayOfString;
begin
  Result := '';
  TempFile := ExpandConstant('{tmp}\gpu_detect.txt');
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

// ── Selected engine helpers (Whisper-only binary) ───────────────────────────
function InstallWhisper: Boolean;
begin
  Result := True;
end;

function InstallCanary: Boolean;
begin
  Result := False;
end;

function SelectedEngineLabel: String;
begin
  Result := 'Whisper (CTranslate2 — ~3 GB VRAM)';
end;

function DefaultEngineName: String;
begin
  Result := 'whisper';
end;

// ── Create Engine Information wizard page ────────────────────────────────────
procedure CreateEngineSelectionPage;
var
  Lbl: TNewStaticText;
  TopPos: Integer;
begin
  EnginePage := CreateCustomPage(wpSelectDir,
    'Speech Engine',
    'CV2T uses the Whisper speech engine for transcription.');

  TopPos := 0;

  // GPU info
  GpuInfoLabel := TNewStaticText.Create(EnginePage);
  GpuInfoLabel.Parent := EnginePage.Surface;
  GpuInfoLabel.Left := 0;
  GpuInfoLabel.Top := TopPos;
  GpuInfoLabel.Width := EnginePage.SurfaceWidth;
  GpuInfoLabel.AutoSize := False;
  GpuInfoLabel.WordWrap := True;

  DetectedGPU := DetectGPU;
  if DetectedGPU <> '' then
    GpuInfoLabel.Caption := 'GPU detected: ' + DetectedGPU
  else
    GpuInfoLabel.Caption := 'No NVIDIA GPU detected — transcription will fall back to CPU (slower).';
  GpuInfoLabel.Height := ScaleY(30);
  TopPos := TopPos + ScaleY(44);

  // Whisper engine info
  Lbl := TNewStaticText.Create(EnginePage);
  Lbl.Parent := EnginePage.Surface;
  Lbl.Left := 0;
  Lbl.Top := TopPos;
  Lbl.Width := EnginePage.SurfaceWidth;
  Lbl.AutoSize := False;
  Lbl.WordWrap := True;
  Lbl.Height := ScaleY(50);
  Lbl.Caption := 'This installer will set up the Whisper engine (CTranslate2, large-v3-turbo). ' +
                  'It provides fast, accurate transcription and requires approximately 3 GB of GPU memory (VRAM). ' +
                  'The model download is about 1 GB.';
  Lbl.Font.Style := [fsBold];
  TopPos := TopPos + ScaleY(64);

  // Canary info header
  Lbl := TNewStaticText.Create(EnginePage);
  Lbl.Parent := EnginePage.Surface;
  Lbl.Left := 0;
  Lbl.Top := TopPos;
  Lbl.Width := EnginePage.SurfaceWidth;
  Lbl.Caption := 'Want higher accuracy? You can add the Canary engine later:';
  TopPos := TopPos + ScaleY(24);

  // Canary method 1
  Lbl := TNewStaticText.Create(EnginePage);
  Lbl.Parent := EnginePage.Surface;
  Lbl.Left := ScaleX(16);
  Lbl.Top := TopPos;
  Lbl.Width := EnginePage.SurfaceWidth - ScaleX(16);
  Lbl.AutoSize := False;
  Lbl.WordWrap := True;
  Lbl.Height := ScaleY(30);
  Lbl.Caption := '1.  Open CV2T, go to Settings, and click "Install Canary Engine".';
  Lbl.Font.Color := clGray;
  TopPos := TopPos + ScaleY(30);

  // Canary method 2
  Lbl := TNewStaticText.Create(EnginePage);
  Lbl.Parent := EnginePage.Surface;
  Lbl.Left := ScaleX(16);
  Lbl.Top := TopPos;
  Lbl.Width := EnginePage.SurfaceWidth - ScaleX(16);
  Lbl.AutoSize := False;
  Lbl.WordWrap := True;
  Lbl.Height := ScaleY(30);
  Lbl.Caption := '2.  Or run the Enable-Canary.ps1 script from the install directory.';
  Lbl.Font.Color := clGray;
end;

// ── "Ready to Install" page customization ───────────────────────────────────
function UpdateReadyMemo(Space, NewLine, MemoUserInfoInfo, MemoDirInfo,
  MemoTypeInfo, MemoComponentsInfo, MemoGroupInfo, MemoTasksInfo: String): String;
var
  Info: String;
begin
  Info := '';

  // Application description
  Info := Info + 'Application:' + NewLine;
  Info := Info + Space + 'CV2T {#MyAppVersion} — Native Windows Voice-to-Text' + NewLine;
  Info := Info + Space + 'Real-time speech transcription using NVIDIA GPUs.' + NewLine;
  Info := Info + NewLine;

  // Install directory
  if MemoDirInfo <> '' then
  begin
    Info := Info + MemoDirInfo + NewLine;
    Info := Info + NewLine;
  end;

  // Selected engine
  Info := Info + 'Speech engine:' + NewLine;
  Info := Info + Space + SelectedEngineLabel + NewLine;
  Info := Info + NewLine;

  // What will be installed
  Info := Info + 'The installer will:' + NewLine;
  Info := Info + Space + '1. Extract CV2T application files' + NewLine;
  Info := Info + Space + '   Includes: cv2t.exe, PySide6 (Qt GUI), CTranslate2,' + NewLine;
  Info := Info + Space + '   sounddevice, numpy, and CUDA runtime libraries' + NewLine;
  Info := Info + Space + '2. Download Whisper model (~1 GB)' + NewLine;
  Info := Info + Space + '3. Create desktop and Start Menu shortcuts' + NewLine;
  Info := Info + Space + '4. Configure Windows Defender exclusions' + NewLine;
  Info := Info + NewLine;

  // GPU info
  if DetectedGPU <> '' then
    Info := Info + 'GPU: ' + DetectedGPU + NewLine
  else
    Info := Info + 'GPU: No NVIDIA GPU detected (will use CPU — slower)' + NewLine;

  Result := Info;
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
          CopyFile(SourcePath, DestPath, False);
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
  OldSettings := ExpandConstant('{userappdata}\CV2T\settings.json');
  NewSettings := ExpandConstant('{app}\config\settings.json');
  if FileExists(OldSettings) and (not FileExists(NewSettings)) then
    CopyFile(OldSettings, NewSettings, False);

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

  OldLogDir := ExpandConstant('{userappdata}\CV2T');
  LogFiles[0] := 'cv2t.log';
  LogFiles[1] := 'cv2t.log.1';
  LogFiles[2] := 'cv2t.log.2';
  for I := 0 to 2 do
  begin
    OldLog := OldLogDir + '\' + LogFiles[I];
    NewLog := ExpandConstant('{app}\logs\') + LogFiles[I];
    if FileExists(OldLog) and (not FileExists(NewLog)) then
      CopyFile(OldLog, NewLog, False);
  end;
end;

// ── Write settings.json with selected engine ────────────────────────────────
procedure WriteDefaultSettings;
var
  SettingsFile: String;
  Json: String;
begin
  SettingsFile := ExpandConstant('{app}\config\settings.json');
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

  DownloadPage := CreateOutputProgressPage('Downloading Model',
    'Downloading the Whisper speech recognition model. This may take several minutes depending on your internet connection.');
  DownloadPage.Show;

  DownloadPage.SetText('Downloading Whisper model (large-v3-turbo, ~1 GB)...',
    'Source: huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo');
  DownloadPage.SetProgress(0, 1);
  try
    Exec(ExePath, 'download-model --engine whisper --target-dir "' + ModelsDir + '"',
         '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
    if ResultCode <> 0 then
      MsgBox('Whisper model download failed (exit code ' + IntToStr(ResultCode) + ').' + #13#10 + #13#10 +
             'You can download it later by running:' + #13#10 +
             '"' + ExePath + '" download-model --engine whisper' + #13#10 + #13#10 +
             'Or from the application: the model will be downloaded on first launch.',
             mbError, MB_OK);
  except
    MsgBox('Could not start Whisper model download.' + #13#10 +
           'You can download models later from the application.',
           mbError, MB_OK);
  end;
  DownloadPage.SetProgress(1, 1);

  DownloadPage.Hide;
end;

// ── Post-install orchestration ──────────────────────────────────────────────
procedure CurStepChanged(CurStep: TSetupStep);
var
  Summary: String;
  InstDir, ModelsDir: String;
  WhisperReady: Boolean;
begin
  if CurStep = ssPostInstall then
  begin
    MigrateOldData;
    WriteDefaultSettings;
    DownloadModels;

    // Build summary text for the summary page
    InstDir := ExpandConstant('{app}');
    ModelsDir := InstDir + '\models';

    Summary := 'CV2T {#MyAppVersion} has been installed successfully.' + #13#10;
    Summary := Summary + '════════════════════════════════════════════' + #13#10 + #13#10;

    Summary := Summary + 'INSTALL LOCATION' + #13#10;
    Summary := Summary + '  ' + InstDir + #13#10 + #13#10;

    Summary := Summary + 'ENGINE SELECTION' + #13#10;
    Summary := Summary + '  ' + SelectedEngineLabel + #13#10;
    Summary := Summary + '  Default engine: ' + DefaultEngineName + #13#10 + #13#10;

    Summary := Summary + 'MODEL STATUS' + #13#10;
    WhisperReady := DirExists(ModelsDir + '\whisper');
    if WhisperReady then
      Summary := Summary + '  [OK] Whisper — downloaded to ' + ModelsDir + '\whisper' + #13#10
    else
      Summary := Summary + '  [!!] Whisper — download failed (run cv2t download-model --engine whisper)' + #13#10;
    Summary := Summary + #13#10;

    Summary := Summary + 'CANARY ENGINE' + #13#10;
    Summary := Summary + '  Not installed. To add Canary later:' + #13#10;
    Summary := Summary + '    - Open CV2T > Settings > Install Canary Engine' + #13#10;
    Summary := Summary + '    - Or run Enable-Canary.ps1 from the install directory' + #13#10;
    Summary := Summary + #13#10;

    Summary := Summary + 'SHORTCUTS' + #13#10;
    Summary := Summary + '  Desktop shortcut created' + #13#10;
    Summary := Summary + '  Start Menu group created' + #13#10 + #13#10;

    Summary := Summary + 'DIRECTORIES' + #13#10;
    Summary := Summary + '  Application:  ' + InstDir + #13#10;
    Summary := Summary + '  Models:       ' + ModelsDir + #13#10;
    Summary := Summary + '  Config:       ' + InstDir + '\config' + #13#10;
    Summary := Summary + '  Logs:         ' + InstDir + '\logs' + #13#10 + #13#10;

    if DetectedGPU <> '' then
      Summary := Summary + 'GPU: ' + DetectedGPU + #13#10
    else
      Summary := Summary + 'GPU: No NVIDIA GPU detected (will use CPU)' + #13#10;
    Summary := Summary + #13#10;

    Summary := Summary + 'SECURITY' + #13#10;
    Summary := Summary + '  Windows Defender exclusions configured for ' + InstDir + #13#10 + #13#10;

    Summary := Summary + 'DEFAULT HOTKEYS' + #13#10;
    Summary := Summary + '  Ctrl+Alt+P   Start recording' + #13#10;
    Summary := Summary + '  Ctrl+Alt+L   Stop recording & transcribe' + #13#10;
    Summary := Summary + '  Ctrl+Alt+Q   Quit application' + #13#10 + #13#10;

    Summary := Summary + 'Hotkeys can be changed in Settings after launching CV2T.';

    SummaryMemo.Text := Summary;
  end;
end;

// ── InitializeWizard: create custom pages ───────────────────────────────────
procedure InitializeWizard;
begin
  CreateEngineSelectionPage;

  // Summary page (shown after install, before the "Finish" page)
  SummaryPage := CreateCustomPage(wpInfoAfter,
    'Installation Summary',
    'Review what was installed and configured.');

  SummaryMemo := TNewMemo.Create(SummaryPage);
  SummaryMemo.Parent := SummaryPage.Surface;
  SummaryMemo.Left := 0;
  SummaryMemo.Top := 0;
  SummaryMemo.Width := SummaryPage.SurfaceWidth;
  SummaryMemo.Height := SummaryPage.SurfaceHeight;
  SummaryMemo.ScrollBars := ssVertical;
  SummaryMemo.ReadOnly := True;
  SummaryMemo.Font.Name := 'Consolas';
  SummaryMemo.Font.Size := 9;
  SummaryMemo.Text := 'Installing...';
end;

// ── Uninstall: prompt for model deletion and clean up remnants ──────────────
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  AppDir, ModelsDir, WhisperDir, CanaryDir: String;
  HasWhisper, HasCanary, DeleteModels: Boolean;
  Msg: String;
begin
  if CurUninstallStep = usUninstall then
  begin
    AppDir := ExpandConstant('{app}');
    ModelsDir := AppDir + '\models';
    WhisperDir := ModelsDir + '\whisper';
    CanaryDir  := ModelsDir + '\canary';
    HasWhisper := DirExists(WhisperDir);
    HasCanary  := DirExists(CanaryDir);
    DeleteModels := False;

    if HasWhisper or HasCanary then
    begin
      if not UninstallSilent then
      begin
        Msg := 'Do you also want to delete the downloaded speech models?' + #13#10 + #13#10;
        if HasWhisper and HasCanary then
          Msg := Msg + '  ' + Chr(8226) + ' Whisper model (~1 GB)' + #13#10 +
                       '  ' + Chr(8226) + ' Canary model (~3 GB)' + #13#10
        else if HasWhisper then
          Msg := Msg + '  ' + Chr(8226) + ' Whisper model (~1 GB)' + #13#10
        else
          Msg := Msg + '  ' + Chr(8226) + ' Canary model (~3 GB)' + #13#10;
        Msg := Msg + #13#10 +
               'Click Yes to remove everything, or No to keep models for a future reinstall.';

        DeleteModels := (MsgBox(Msg, mbConfirmation, MB_YESNO) = IDYES);
      end;
      // Silent uninstall: keep models by default
    end;

    if DeleteModels then
    begin
      if HasWhisper then
        DelTree(WhisperDir, True, True, True);
      if HasCanary then
        DelTree(CanaryDir, True, True, True);
    end;

    // Remove models dir if now empty
    if DirExists(ModelsDir) then
      RemoveDir(ModelsDir);
  end;

  if CurUninstallStep = usPostUninstall then
  begin
    // Clean up the app directory if it is empty after standard uninstall
    AppDir := ExpandConstant('{app}');
    if DirExists(AppDir) then
      RemoveDir(AppDir);
  end;
end;
