const { app, BrowserWindow, Menu, ipcMain, dialog, shell, nativeTheme } = require('electron')
const contextMenu = require('electron-context-menu');
const osu = require('node-os-utils')
const path = require("path");
const writeYamlFile = require('write-yaml-file')
const fs = require('fs')
const os = require('os')

const { handleGetSingleFolder,handleGetMultipleFolders, handleGetSingleFile, handleGetMultipleFiles, handleGetMultiple } = require('./modules/dialogHandler')
const { discoverWorkflows, workflowToConfig } = require('./modules/workflows')
const { CondaEnvironment} = require('./modules/cmd');
const { buildMenu } = require('./modules/menu')
const { Profile } = require('./modules/profile')

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
let environment;
let profile;

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
    
    environment = new CondaEnvironment(profile)
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

function handleStartWorkflow(workflow) {

    const workflowFolder = workflow.output.path
    const config = workflowToConfig(workflow)

    // check if workflow folder exists
    if (!fs.existsSync(workflowFolder)) {
        dialog.showMessageBox(mainWindow, {
            type: 'error',
            title: 'Workflow Failed to Start',
            message: `Could not start workflow. Output folder ${workflowFolder} does not exist.`,
        }).catch((err) => {
            console.log(err)
        })
        return Promise.resolve("Workflow failed to start.")
    }

    const configPath = path.join(workflowFolder, "config.yaml")
    // save config.yaml in workflow folder
    writeYamlFile.sync(configPath, config, {lineWidth: -1})
    
    return environment.spawn(`conda run -n ${profile.config.conda.envName} --no-capture-output alphadia extract --config ${configPath}`)
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.whenReady().then(() => {
    
    console.log(app.getLocale())
    console.log(app.getSystemLocale())
    createWindow(); 

    ipcMain.handle('get-single-folder', handleGetSingleFolder(mainWindow))
    ipcMain.handle('get-multiple-folders', handleGetMultipleFolders(mainWindow))
    ipcMain.handle('get-single-file', handleGetSingleFile(mainWindow))
    ipcMain.handle('get-multiple-files', handleGetMultipleFiles(mainWindow))
    ipcMain.handle('get-multiple', handleGetMultiple(mainWindow))
    ipcMain.handle('get-utilisation', handleGetUtilisation)
    ipcMain.handle('get-workflows', () => workflows)

    ipcMain.handle('get-environment', () => environment.getEnvironmentStatus())

    ipcMain.handle('run-command', (event, cmd) => environment.spawn(cmd))
    ipcMain.handle('get-output-rows', (event, {limit, offset}) => environment.getOutputRows(limit, offset))
    ipcMain.handle('get-output-length', () => environment.getOutputLength())
    
    ipcMain.handle('start-workflow', (event, workflow) => handleStartWorkflow(workflow))
    ipcMain.handle('abort-workflow', () => environment.kill())

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