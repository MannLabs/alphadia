; Script generated by the Inno Setup Script Wizard.
; SEE THE DOCUMENTATION FOR DETAILS ON CREATING INNO SETUP SCRIPT FILES!

; Note: apparently, ISCC uses the directory of the .iss input file as the working directory,
; so all paths are given relative to the location of this .iss file.

#define MyAppName "AlphaDIA"
#define MyAppVersion "1.9.3-dev5"
#define MyAppPublisher "Max Planck Institute of Biochemistry, Mann Labs"
#define MyAppURL "https://github.com/MannLabs/alphadia"
#define MyAppExeName "alphadia-gui.exe"

[Setup]
; NOTE: The value of AppId uniquely identifies this application. Do not use the same AppId value in installers for other applications.
; (To generate a new GUID, click Tools | Generate GUID inside the IDE.)
AppId={{alphadia_Mann_Labs_MPI}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
;AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DisableProgramGroupPage=yes
LicenseFile=..\..\LICENSE.txt
; Uncomment the following line to run in non administrative install mode (install for current user only.)
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
; release workflow expects artifact at root of repository
OutputDir=..\..\
; example for BUILD_NAME: alphadia-1.9.3-dev5-win-x64
OutputBaseFilename={#GetEnv('BUILD_NAME')}
SetupIconFile=..\logos\alphadia.ico
Compression=lzma
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\..\gui\out\alphadia-gui-win32-x64\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\gui\out\alphadia-gui-win32-x64\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\..\dist_pyinstaller\alphadia\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; NOTE: Don't use "Flags: ignoreversion" on any shared system files

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
