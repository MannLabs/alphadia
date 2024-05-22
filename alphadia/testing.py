import os
import base64
from urllib.request import urlopen
from urllib.request import urlretrieve
import cgi
import zipfile
import progressbar
import logging
import typing


class Progress:  # pragma: no cover
    """Class to report the download progress of a file to the console."""

    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


def _encode_url_onedrive(sharing_url: str) -> str:  # pragma: no cover
    """Encode onedrive sharing link as url for downloading files.

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


def _get_filename_onedrive(sharing_url: str) -> str:  # pragma: no cover
    """Get filename from onedrive sharing link.

    Parameters
    ----------
    sharing_url : str
        onedrive sharing link

    Returns
    -------
    filename : str
        filename of the file shared cia onedrive

    """

    encoded_url = _encode_url_onedrive(sharing_url)

    try:
        remotefile = urlopen(encoded_url)
    except:
        print(f"Could not open {sharing_url} for reading filename")
        raise ValueError(f"Could not open {sharing_url} for reading filename") from None

    info = remotefile.info()["Content-Disposition"]
    value, params = cgi.parse_header(info)
    return params["filename"]


def _download_onedrive(
    sharing_url: str, output_dir: str
) -> typing.Union[str, None]:  # pragma: no cover
    """Download file from onedrive sharing link.

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
    filename = _get_filename_onedrive(sharing_url)
    encoded_url = _encode_url_onedrive(sharing_url)

    output_path = os.path.join(output_dir, filename)
    try:
        path, message = urlretrieve(encoded_url, output_path, Progress())
        print(f"{filename} successfully downloaded")
        return path
    except:
        print(f"Could not download {filename} from onedrive")
        return None


def update_onedrive(
    sharing_url: str, output_dir: str, unzip: bool = True
) -> None:  # pragma: no cover
    """Download file from onedrive sharing link if it does not yet exist.

    Parameters
    ----------
    sharing_url : str
        onedrive sharing link

    output_dir : str
        path to output directory

    unzip : bool
        unzip file if it ends with .zip

    """

    filename = _get_filename_onedrive(sharing_url)
    output_path = os.path.join(output_dir, filename)
    if not os.path.exists(output_path):
        print(f"{filename} does not yet exist")
        _download_onedrive(sharing_url, output_dir)

        # if file ends with .zip and zip=True
        if unzip and filename.endswith(".zip"):
            print(f"Unzipping {filename}")
            with zipfile.ZipFile(output_path, "r") as zip_ref:
                zip_ref.extractall(output_dir)
            print(f"{filename} successfully unzipped")

    else:
        print(f"{filename} already exists")


def _encode_url_datashare(sharing_url: str) -> str:  # pragma: no cover
    """Encode datashare sharing link as url for downloading files.

    Parameters
    ----------
    sharing_url : str
        onedrive sharing link

    Returns
    -------
    encoded_url : str
        encoded url for directly downloading files

    """

    if "/download?" not in sharing_url:
        return f"{sharing_url}/download"

    return sharing_url


def _get_filename_datashare(sharing_url: str, tar=False) -> str:  # pragma: no cover
    """Fet filename from datashare sharing link.

    Parameters
    ----------
    sharing_url : str
        onedrive sharing link

    Returns
    -------
    filename : str
        filename of the file shared cia onedrive

    """

    encoded_url = _encode_url_datashare(sharing_url)

    try:
        remotefile = urlopen(encoded_url)
    except:
        raise ValueError(f"Could not open {sharing_url} for reading filename") from None

    info = remotefile.info()["Content-Disposition"]
    value, params = cgi.parse_header(info)
    filename = params["filename"]
    return filename


def _download_from_datashare(
    sharing_url: str, output_dir: str
) -> typing.Union[str, None]:  # pragma: no cover
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
    filename = _get_filename_datashare(sharing_url)
    encoded_url = _encode_url_datashare(sharing_url)

    output_path = os.path.join(output_dir, filename)

    try:
        path, message = urlretrieve(encoded_url, output_path, Progress())
        print(f"{filename} successfully downloaded")

        return path
    except Exception:
        print(f"Could not download {filename} from datashare")

        return None


def update_datashare(
    sharing_url: str, output_dir: str, force_download=False
) -> str:  # pragma: no cover
    """download file from datashare sharing link if it does not yet exist

    Parameters
    ----------
    sharing_url : str
        datashare sharing link

    output_dir : str
        path to output directory

    unzip : bool
        unzip file if it ends with .zip

    force_download : bool
        force download even if file already exists

    """

    filename = _get_filename_datashare(sharing_url)
    output_path = os.path.join(output_dir, filename)
    unzipped_path = os.path.join(output_dir, filename.replace(".zip", ""))

    if not os.path.exists(unzipped_path):
        print(f"{filename} does not yet exist")
        _download_from_datashare(sharing_url, output_dir)

        _handle_archive(filename, output_dir, output_path)

    elif force_download:
        print(f"{filename} already exists, but force_download=True")

        try:
            os.remove(output_path)
            print(f"{filename} successfully removed")
        except Exception as e:
            logging.error(f"Could not remove {filename}")
            return

        _download_from_datashare(sharing_url, output_dir)

        _handle_archive(filename, output_dir, output_path)
    else:
        print(f"{filename} already exists")

    return unzipped_path


def _handle_archive(filename: str, output_dir: str, output_path: str) -> None:
    """Unpack archive and remove it."""
    if filename.endswith(".zip"):
        with zipfile.ZipFile(output_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"{filename} successfully unzipped")
        os.remove(output_path)
