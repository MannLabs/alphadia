# native imports
import os
import base64
from urllib.request import urlopen
from urllib.request import urlretrieve
import cgi
import zipfile
import progressbar
import logging
import typing

# alphadia imports

# alpha family imports

# third party imports

class Progress(): # pragma: no cover
    """Class to report the download progress of a file to the console.
    """
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar=progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()

def encode_url_onedrive(sharing_url: str) -> str: # pragma: no cover
    """encode onedrive sharing link as url for downloading files

    Parameters
    ----------
    sharing_url : str
        onedrive sharing link

    Returns
    -------
    encoded_url : str
        encoded url for downloading files

    """
    
    b64_string = base64.urlsafe_b64encode(str.encode(sharing_url)).decode("utf-8")
    encoded_url = f'https://api.onedrive.com/v1.0/shares/u!{b64_string.replace("=", "")}/root/content'
    return encoded_url

def filename_onedrive(sharing_url: str) -> str: # pragma: no cover
    """get filename from onedrive sharing link

    Parameters
    ----------  
    sharing_url : str
        onedrive sharing link

    Returns
    -------
    filename : str
        filename of the file shared cia onedrive

    """

    encoded_url = encode_url_onedrive(sharing_url)

    try:
        remotefile = urlopen(encoded_url)
    except:
        logging.info(f'Could not open {sharing_url} for reading filename')
        raise ValueError(f'Could not open {sharing_url} for reading filename') from None
    
    info = remotefile.info()['Content-Disposition']
    value, params = cgi.parse_header(info)
    return params["filename"]

def download_onedrive(sharing_url: str, output_dir: str) -> typing.Union[str, None]: # pragma: no cover
    """download file from onedrive sharing link

    Parameters
    ----------
    sharing_url : str
        onedrive sharing link

    output_dir : str
        path to output directory

    Returns
    -------
    path : str
        local path to downloaded file

    """
    filename = filename_onedrive(sharing_url)
    encoded_url = encode_url_onedrive(sharing_url)

    output_path = os.path.join(output_dir, filename)
    try:
        path, message = urlretrieve(encoded_url, output_path, Progress())
        logging.info(f'{filename} successfully downloaded')
        return path
    except:
        logging.info(f'Could not download {filename} from onedrive')
        return None

def update_onedrive(sharing_url: str, output_dir: str, unzip: bool = True) -> None: # pragma: no cover
    """download file from onedrive sharing link if it does not yet exist

    Parameters
    ----------
    sharing_url : str
        onedrive sharing link

    output_dir : str
        path to output directory

    unzip : bool
        unzip file if it ends with .zip

    """
    
    filename = filename_onedrive(sharing_url)
    output_path = os.path.join(output_dir, filename)
    if not os.path.exists(output_path):
        logging.info(f'{filename} does not yet exist')
        download_onedrive(sharing_url, output_dir)

        # if file ends with .zip and zip=True
        if unzip and filename.endswith('.zip'):
            logging.info(f'Unzipping {filename}')
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            logging.info(f'{filename} successfully unzipped')

    else:
        logging.info(f'{filename} already exists')

def encode_url_datashare(sharing_url: str) -> str: # pragma: no cover
    """encode datashare sharing link as url for downloading files

    Parameters
    ----------
    sharing_url : str
        onedrive sharing link

    Returns
    -------
    encoded_url : str
        encoded url for directly downloading files

    """
    
    encoded_url = f'{sharing_url}/download'
    return encoded_url


def filename_datashare(sharing_url: str, tar=False) -> str: # pragma: no cover
    """get filename from onedrive sharing link

    Parameters
    ----------  
    sharing_url : str
        onedrive sharing link

    Returns
    -------
    filename : str
        filename of the file shared cia onedrive

    """

    encoded_url = encode_url_datashare(sharing_url)

    try:
        remotefile = urlopen(encoded_url)
    except:
        #logging.info(f'Could not open {sharing_url} for reading filename')
        raise ValueError(f'Could not open {sharing_url} for reading filename') from None
    
    info = remotefile.info()['Content-Disposition']
    value, params = cgi.parse_header(info)
    filename = params["filename"]
    return filename

def download_datashare(sharing_url: str, output_dir: str) -> typing.Union[str, None]: # pragma: no cover
    """download file from datashare sharing link

    Parameters
    ----------
    sharing_url : str
        onedrive sharing link

    output_dir : str
        path to output directory

    Returns
    -------
    path : str
        local path to downloaded file

    """
    filename = filename_datashare(sharing_url)
    encoded_url = encode_url_datashare(sharing_url)

    output_path = os.path.join(output_dir, filename)

    try:
        path, message = urlretrieve(encoded_url, output_path, Progress())
        logging.info(f'{filename} successfully downloaded')

        return path
    except Exception as e:
        logging.info(f'Could not download {filename} from datashare')

        return None
    
def update_datashare(sharing_url: str, output_dir: str, force=False) -> None: # pragma: no cover
    """download file from datashare sharing link if it does not yet exist

    Parameters
    ----------
    sharing_url : str
        datashare sharing link

    output_dir : str
        path to output directory

    unzip : bool
        unzip file if it ends with .zip

    force : bool
        force download even if file already exists

    """

    
    filename = filename_datashare(sharing_url)
    output_path = os.path.join(output_dir, filename)
    unzipped_path = os.path.join(output_dir, filename.replace('.zip', ''))

    # file does not yet exist
    if not os.path.exists(unzipped_path):
        logging.info(f'{filename} does not yet exist')
        download_datashare(sharing_url, output_dir)

        # if file ends with .zip and zip=True
        if filename.endswith('.zip'):
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            logging.info(f'{filename} successfully unzipped')
            os.remove(output_path)

    # file already exists
    else:
        # force download
        if force:
            logging.info(f'{filename} already exists, but force=True')

            # remove file
            try:
                os.remove(output_path)
                logging.info(f'{filename} successfully removed')
            except:
                logging.error(f'Could not remove {filename}')
                return

            download_datashare(sharing_url, output_dir)

            # if file ends with .zip and zip=True
            if filename.endswith('.zip'):
                with zipfile.ZipFile(output_path, 'r') as zip_ref:
                    zip_ref.extractall(output_dir)
                logging.info(f'{filename} successfully unzipped')
                os.remove(output_path)

        logging.info(f'{filename} already exists')