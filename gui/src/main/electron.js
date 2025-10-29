const { app, BrowserWindow, Menu, ipcMain, dialog, shell, nativeTheme } = require('electron')
const contextMenu = require('electron-context-menu');
const osu = require('node-os-utils')
const path = require("path");
const writeYamlFile = require('write-yaml-file')
const fs = require('fs')
const os = require('os')

const { handleGetSingleFolder,handleGetMultipleFolders, handleGetSingleFile, handleGetMultipleFiles, handleGetMultiple } = require('./modules/dialogHandler')
const { discoverWorkflows, workflowToConfig } = require('./modules/workflows')
const { ExecutionManager } = require('./modules/engine')
const { buildMenu } = require('./modules/menu')
const { Profile } = require('./modules/profile')

// Set encoding for Windows to handle UTF-8 properly
if (process.platform === 'win32') {
  process.stdout.setEncoding('utf8');
  process.env.PYTHONIOENCODING = 'utf-8';
}

contextMenu({
	showSaveImageAs: false,
    showCopyLink: false,
    showInspectElement: false,
    showSearchWithGoogle: false,
    showLookUpSelection: false,
});

// Handle creating/removing shortcuts on Windows when installing/uninstalling.
if (require('electron-squirrel-startup')) app.quit();

let mainWindow;
let workflows;
let profile;
let executionManager;

function createWindow() {
  // Create the browser window.
  mainWindow = new BrowserWindow({
    width: 900,
    height: 600,
    minWidth: 900,
    minHeight: 600,
    title: "alphaDIA",
    webPreferences: {
        nodeIntegration: false, // is default value after Electron v5
        contextIsolation: true, // protect against prototype pollution
        enableRemoteModule: false, // turn off remote
        preload: path.join(__dirname, 'preload.js')
    },
  });
    mainWindow.setTitle("alphaDIA");
    mainWindow.loadFile(path.join(__dirname, "../dist/index.html"));

    profile = new Profile()

    Menu.setApplicationMenu(buildMenu(mainWindow, profile));

    // Open the DevTools if NODE_ENV=dev
    if (process.env.NODE_ENV === "dev"){
        mainWindow.webContents.openDevTools({ mode: "detach" });
    }

    executionManager = new ExecutionManager(profile)
    workflows = discoverWorkflows(mainWindow)
}

handleOpenLink = (event, url) => {
    event.preventDefault()
    shell.openExternal(url)
}

async function handleGetUtilisation (event) {

    var mem = osu.mem

    const values = await Promise.all([osu.cpu.usage(), mem.info()])
    const cpu = values[0]
    const memory = values[1]
    return {
        ...memory,
        cpu
    }
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.whenReady().then(() => {

    console.log(app.getLocale())
    console.log(app.getSystemLocale())
    createWindow();

    ipcMain.handle('get-engine-status', () => executionManager.getEngineStatus())

    ipcMain.handle('start-workflow-new', (event, workflow, engineIdx) => executionManager.startWorkflow(workflow, engineIdx))
    ipcMain.handle('abort-workflow-new', (event, runIdx) => executionManager.abortWorkflow(runIdx))

    ipcMain.handle('get-output-length-new', (event, runIdx) => executionManager.getOutputLength(runIdx))
    ipcMain.handle('get-output-rows-new', (event, runIdx, {limit, offset}) => executionManager.getOutputRows(runIdx, limit, offset))

    ipcMain.handle('get-single-folder', handleGetSingleFolder(mainWindow))
    ipcMain.handle('get-multiple-folders', handleGetMultipleFolders(mainWindow))
    ipcMain.handle('get-single-file', handleGetSingleFile(mainWindow))
    ipcMain.handle('get-multiple-files', handleGetMultipleFiles(mainWindow))
    ipcMain.handle('get-multiple', handleGetMultiple(mainWindow))
    ipcMain.handle('get-utilisation', handleGetUtilisation)
    ipcMain.handle('get-workflows', () => workflows)

    ipcMain.on('open-link', handleOpenLink)
    nativeTheme.on('updated', () => {
        console.log("Theme changed to: " + nativeTheme)
        mainWindow.webContents.send('theme-change', nativeTheme.shouldUseDarkColors)
    })

    powerMonitor.on("lock-screen", () => {
        powerSaveBlocker.start("prevent-display-sleep");
    });
    powerMonitor.on("suspend", () => {
        powerSaveBlocker.start("prevent-app-suspension");
    });

});

app.on('window-all-closed', () => {
    app.quit();
});
