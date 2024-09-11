# Powershell script to build the GUI for Windows

cd gui
# delete old build using powershell
rm -rf ./out
rm -rf  ./dist

npm install
npm run make

cd ..
