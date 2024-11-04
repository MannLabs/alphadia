# Build the install package for Windows.
# This script must be run from the root of the repository after running build_installer_windows.ps1
&  "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" .\release\windows\alphadia_innoinstaller.iss
