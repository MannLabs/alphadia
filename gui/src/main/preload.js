const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('electronAPI', {
    getSingleFolder: () => ipcRenderer.invoke('get-single-folder'),
    getMultipleFolders: () => ipcRenderer.invoke('get-multiple-folders'),
    getSingleFile: () => ipcRenderer.invoke('get-single-file'),
    getMultipleFiles: () => ipcRenderer.invoke('get-multiple-files'),
    getMultiple: () => ipcRenderer.invoke('get-multiple'),
    getUtilisation: () => ipcRenderer.invoke('get-utilisation'),
    getWorkflows: () => ipcRenderer.invoke('get-workflows'),

    getEnvironment: () => ipcRenderer.invoke('get-environment'),

    getOutput: () => ipcRenderer.invoke('get-output'),
    getOutputRows: ({limit, offset}) => ipcRenderer.invoke('get-output-rows', {limit, offset}),
    getOutputLength: () => ipcRenderer.invoke('get-output-length'),

    openLink: (url) => ipcRenderer.send('open-link', url),
    runCommand: (command) => ipcRenderer.invoke('run-command', command),
    startWorkflow: (workflow) => ipcRenderer.invoke('start-workflow', workflow),
    abortWorkflow: () => ipcRenderer.invoke('abort-workflow'),

    onStdoutData: (callback) => ipcRenderer.on('stdout-data', callback),
    onStderrData: (callback) => ipcRenderer.on('stderr-data', callback),
    onCloseCode: (callback) => ipcRenderer.on('close-code', callback),

    onThemeChange: (callback) => ipcRenderer.on('theme-change', callback)
})