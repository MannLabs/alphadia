const {app, Menu} = require('electron')
const path = require('path')
const {shell} = require('electron')

function buildMenu(mainWindow, profile) {

    const isMac = process.platform === 'darwin'

    const template = [
    ...(isMac
        ? [{
            label: app.name,
            submenu: [
            { role: 'about' },
            { type: 'separator' },
            { id: 'preferences', label: 'Preferences', click: () => { profile.openProfile() } },
            { type: 'separator' },
            { role: 'hide' },
            { role: 'hideOthers' },
            { role: 'unhide' },
            { type: 'separator' },
            { role: 'quit' }
            ]
        }]
        : []),
    {
        label: 'File',
        submenu: [
        isMac ? { role: 'close' } : { role: 'quit' },
        ...(isMac ? []: [{ id: 'preferences', label: 'Preferences', click: () => { profile.openProfile() } }])
        ]
    },
    {
        label: 'Window',
        submenu: [
        { role: 'minimize' },
        { role: 'zoom' },
        ...(isMac
            ? [
                { type: 'separator' },
                { role: 'front' },
                { type: 'separator' },
                { role: 'window' }
            ]
            : [
                { role: 'close' }
            ])
        ]
    },
    {
        role: 'help',
        submenu: [
        {
            label: 'Learn More',
            click: async () => {
            const { shell } = require('electron')
            await shell.openExternal('https://electronjs.org')
            }
        }
        ]
    }
    ]

    return Menu.buildFromTemplate(template)
}

module.exports = {
    buildMenu
}