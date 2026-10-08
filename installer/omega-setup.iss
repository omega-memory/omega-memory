; OMEGA Memory - Windows Installer (Inno Setup 6)
;
; Bundles Python 3.12 embeddable, installs omega-memory[server] via pip,
; and auto-configures Claude Desktop's MCP config.
;
; Build: iscc /DMyAppVersion=1.5.20 omega-setup.iss (or via GitHub Actions)
; Requires: build\python\ (embeddable Python), build\get-pip.py

#define MyAppName "OMEGA Memory"
; The omega-memory release this installer pip-installs on the user's PC. The
; workflow passes the tag's version; there is no default, because a pinned
; default kept every installer from v1.5.4 to v1.5.19 on Core 1.5.4.
#ifndef MyAppVersion
  #error Pass the omega-memory version: iscc /DMyAppVersion=X.Y.Z omega-setup.iss
#endif
#define MyAppPublisher "OMEGA Memory Maintainers"
#define MyAppURL "https://omegamax.co"

[Setup]
AppId={{E8A3F2D1-7B4C-4E9F-A6D8-3C1F5B2E9A70}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
DefaultDirName={localappdata}\OMEGA
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=dist
OutputBaseFilename=omega-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\python\python.exe
; Code signing — enabled by passing /DSIGN and /Ssigntool=... to ISCC in CI.
; When SIGN is defined, Inno Setup will sign the uninstaller exe embedded in the installer.
#ifdef SIGN
SignTool=signtool
SignToolRetryCount=3
SignToolRetryDelay=2000
#endif

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
; Bundled Python 3.12 embeddable
Source: "build\python\*"; DestDir: "{app}\python"; Flags: ignoreversion recursesubdirs

; get-pip.py for bootstrapping pip
Source: "build\get-pip.py"; DestDir: "{app}"; Flags: ignoreversion

; Post-install configuration script
Source: "configure_claude.py"; DestDir: "{app}"; Flags: ignoreversion

; Repair batch file (user-facing fix if auto-config fails)
Source: "repair-config.bat"; DestDir: "{app}"; Flags: ignoreversion

[Run]
; Step 1: Enable site-packages in embedded Python (uncomment import line in ._pth file)
Filename: "powershell.exe"; \
  Parameters: "-NoProfile -Command ""$f = Get-ChildItem '{app}\python\python*._pth' | Select-Object -First 1; (Get-Content $f.FullName) -replace '^#import site','import site' | Set-Content $f.FullName"""; \
  StatusMsg: "Configuring Python..."; \
  Flags: runhidden waituntilterminated

; Step 2: Install pip
Filename: "{app}\python\python.exe"; \
  Parameters: """{app}\get-pip.py"" --no-warn-script-location"; \
  StatusMsg: "Installing pip..."; \
  Flags: runhidden waituntilterminated

; Step 3: Install omega-memory with server dependencies
Filename: "{app}\python\python.exe"; \
  Parameters: "-m pip install omega-memory[server]=={#MyAppVersion} --no-warn-script-location"; \
  StatusMsg: "Installing OMEGA Memory (this may take a minute)..."; \
  Flags: runhidden waituntilterminated

; Step 4: Configure Claude Desktop
Filename: "{app}\python\python.exe"; \
  Parameters: """{app}\configure_claude.py"" --install-dir ""{app}"""; \
  StatusMsg: "Configuring Claude Desktop..."; \
  Flags: runhidden waituntilterminated

[Icons]
; Start Menu shortcut for repair
Name: "{group}\Repair OMEGA Config"; Filename: "{app}\repair-config.bat"; \
  Comment: "Fix OMEGA connection to Claude Desktop"

[UninstallRun]
; Remove OMEGA from Claude Desktop config before uninstalling
Filename: "{app}\python\python.exe"; \
  Parameters: """{app}\configure_claude.py"" --uninstall"; \
  Flags: runhidden waituntilterminated

[UninstallDelete]
; Clean up pip cache and __pycache__ dirs
Type: filesandordirs; Name: "{app}\python\Lib"
Type: filesandordirs; Name: "{app}\python\Scripts"

[Messages]
WelcomeLabel2=This will install {#MyAppName} for Claude Desktop.%n%nOMEGA gives Claude persistent memory across conversations.%n%nRequirements:%n- Claude Desktop must be installed%n- Internet connection (for downloading the embedding model on first use)
FinishedLabel=Setup has installed {#MyAppName} for Claude Desktop.%n%nRestart Claude Desktop to start using OMEGA. Say "hello" to Claude and OMEGA's memory tools will be available.

[Code]
function ClaudeConfigHasOmega(): Boolean;
var
  ConfigPath: String;
  Content: AnsiString;
begin
  Result := False;
  ConfigPath := ExpandConstant('{%APPDATA}\Claude\claude_desktop_config.json');
  if FileExists(ConfigPath) then
  begin
    if LoadStringFromFile(ConfigPath, Content) then
    begin
      Result := Pos('"omega-memory"', Content) > 0;
    end;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    if not ClaudeConfigHasOmega() then
    begin
      MsgBox(
        'OMEGA was installed but could not configure Claude Desktop automatically.' + #13#10 + #13#10 +
        'To fix this:' + #13#10 +
        '1. Open Start Menu and find "Repair OMEGA Config"' + #13#10 +
        '2. Run it and follow the instructions' + #13#10 + #13#10 +
        'If that doesn''t work, check the log file at:' + #13#10 +
        ExpandConstant('{app}\configure_claude.log'),
        mbInformation, MB_OK
      );
    end;
  end;
end;
