"""Download files from sharing links."""

import base64
import cgi
import os
import traceback
import zipfile
from abc import ABC, abstractmethod
from urllib.request import urlopen, urlretrieve

import progressbar


class Progress:  # pragma: no cover
    """Class to report the download progress of a file to the console."""

    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if total_size < 0:
            # disable progress when the downloaded item is a directory
            return

        if not self.pbar:
            self.pbar = progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


class FileDownloader(ABC):
    """Abstract class for downloading files from sharing links."""

    def __init__(self, url: str, output_dir: str):
        """Initialize FileDownloader."""
        self._url = url
        self._output_dir = output_dir
        self._encoded_url = self._encode_url()

        self._file_name = self._get_filename()
        self._is_archive = self._file_name.endswith(".zip")

        self._output_path = os.path.join(output_dir, self._file_name)
        self._unzipped_output_path = (
            self._output_path.replace(".zip", "")
            if self._is_archive
            else self._output_path
        )

    @abstractmethod
    def _encode_url(self) -> None:
        pass

    def download(self) -> str:  # pragma: no cover
        """Download file from sharing link if it does not yet exist and return its location."""

        if not os.path.exists(self._unzipped_output_path):
            print(f"{self._unzipped_output_path} does not yet exist")
            self._download_file()

            self._handle_archive()

        else:
            print(f"{self._unzipped_output_path} already exists")

        return self._unzipped_output_path

    def _get_filename(self) -> str:  # pragma: no cover
        """Get filename from url."""
        try:
            remotefile = urlopen(self._encoded_url)
        except Exception:
            print(f"Could not open {self._url} for reading filename")
            raise ValueError(
                f"Could not open {self._url} for reading filename"
            ) from None

        info = remotefile.info()["Content-Disposition"]
        value, params = cgi.parse_header(info)
        return params["filename"]

    def _download_file(self) -> None:  # pragma: no cover
        """Download file from sharing link.

        Returns
        -------
        path : str
            local path to downloaded file

        """
        try:
            path, message = urlretrieve(
                self._encoded_url, self._output_path, Progress()
            )
            print(f"{self._file_name} successfully downloaded to {path}")

        except Exception as e:
            print(f"{e} {traceback.print_exc()}")
            raise ValueError(f"Could not download {self._file_name}: {e}") from e

    def _handle_archive(self) -> None:
        """Unpack archive and remove it."""
        if self._is_archive:
            with zipfile.ZipFile(self._output_path, "r") as zip_ref:
                zip_ref.extractall(self._output_dir)
            print(f"{self._file_name} successfully unzipped")
            os.remove(self._output_path)


class OnedriveDownloader(FileDownloader):
    """Class for downloading files from onedrive sharing links."""

    def _encode_url(self) -> str:  # pragma: no cover
        """Encode onedrive sharing link as url for downloading files."""

        b64_string = base64.urlsafe_b64encode(str.encode(self._url)).decode("utf-8")
        encoded_url = f'https://api.onedrive.com/v1.0/shares/u!{b64_string.replace("=", "")}/root/content'
        return encoded_url


class DataShareDownloader(FileDownloader):
    """Class for downloading files from datashare sharing links."""

    def _encode_url(self) -> str:  # pragma: no cover
        """Encode datashare sharing link as url for downloading files."""

        # this is the case if the url points to a folder
        if "/download?" not in self._url:
            return f"{self._url}/download"

        return self._url
