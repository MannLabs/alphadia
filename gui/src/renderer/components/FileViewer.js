import { DataGrid } from "@mui/x-data-grid";

const columns = [
    { field: 'id', headerName: 'Index', width: 50 },
    { field: 'fileExtension', headerName: 'Type', width: 100},
    { field: 'folderName', headerName: 'Folder', width: 100, flex: 0.5 },
    { field: 'fileName', headerName: 'File Name', width:100, flex: 0.5 },
];

function getFileExtension(fileName) {
    var dotIndex = fileName.lastIndexOf('.');
    if (dotIndex === -1 || dotIndex === fileName.length - 1) {
      return ''; // No extension or dot at the end of the filename
    } else {
      return fileName.substring(dotIndex + 1);
    }
  }

const FileViewer = ({
    label = "File",
    active = true,
    path = [],
    tooltipText = "",
    onChange = () => {},
    ...props
    }) => {

    const rows = path.map((file, index) => {

        // eslint-disable-next-line no-useless-escape
        const fileName = file.replace(/^.*[\\\/]/, '')
        const fileExtension = getFileExtension(fileName)

        return {
            id: index,
            fileName: fileName,
            // eslint-disable-next-line no-useless-escape
            folderName: file.replace(/[^\/]*$/, ''),
            fileExtension: fileExtension
        }
    })

    return (
    <DataGrid
        rows={rows}
        columns={columns}
        pageSize={5}
        rowHeight={40}
        density="compact"
        />
)}

export default FileViewer;
