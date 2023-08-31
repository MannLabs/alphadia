module.exports = {
  packagerConfig: {
    asar: true,
    icon: 'assets/alphadia',
    executableName: 'alphadia-gui',
  },
  rebuildConfig: {},
  makers: [
    {
      name: '@electron-forge/maker-zip',
      platforms: ['darwin', 'win32','linux'],
    },
  ],
  plugins: [
    {
      name: '@electron-forge/plugin-auto-unpack-natives',
      config: {},
    },
  ],
};
