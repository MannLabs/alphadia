# Powershell script to build the GUI for Windows

cd gui
# delete old build using powershell
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue ./out
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue ./dist


npm install
npm run make

cd ..
