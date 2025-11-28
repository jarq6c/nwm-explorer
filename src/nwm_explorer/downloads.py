"""Utilities to support downloading files."""
from typing import Callable
from time import sleep
from pathlib import Path
import asyncio
import ssl
from http import HTTPStatus
import warnings
import inspect
from yarl import URL
import aiohttp
from aiohttp.typedefs import LooseHeaders
import aiofiles

from .logger import get_logger

class FileValidationError(Exception):
    """Exception raised when file fails to validate."""

def default_file_validator(filepath: Path) -> None:
    """
    Validate that given filepath opens and closes without raising.

    Parameters
    ----------
    filepath: Path
        Path to file.
    
    Returns
    -------
    None

    Raises
    ------
    FileValidationError
    """
    try:
        with filepath.open("rb"):
            return
    except Exception as e:
        raise FileValidationError(e) from e

async def download_file_awaitable(
        url: str | URL,
        filepath: str | Path,
        session: aiohttp.ClientSession,
        chunk_size: int = 1024,
        overwrite: bool = False,
        ssl_context: ssl.SSLContext | None = None,
        timeout: int = 300,
        retries: int = 3
        ) -> None:
    """
    Retrieve a single file from url and save to filepath.
    
    Parameters
    ----------
    url: str | URL, required
        Source URL to download.
    filepath: str | Path, required
        Destination to save file.
    session: ClientSession, required
        Session object used for retrieval.
    chunk_size: int, optional, default 1024
        Amount of data to write at a time (KB).
    overwrite: bool, optional, default False
        If filepath exists, overwrite if True, else skip download.
    ssl_context: SSLContext, optional
        SSL configuration object. Uses system defaults unless otherwise
        specified.
    timeout: int, optional, default 300
        Maximum number of seconds to wait for a response to return.
    retries: int, optional, default 3
        Number of download attempts in the event of failure.
        
    Returns
    -------
    None
    """
    # Check for file existence
    if not overwrite and Path(filepath).exists():
        message = f"File exists, skipping download of {filepath}"
        warnings.warn(message, UserWarning)
        return

    # SSL
    if ssl_context is None:
        ssl_context = ssl.create_default_context()

    # Fetch
    for attempt in range(retries):
        try:
            async with session.get(url, ssl=ssl_context, timeout=timeout,
                raise_for_status=True) as response:
                # Stream download
                async with aiofiles.open(filepath, 'wb') as fo:
                    while True:
                        chunk = await response.content.read(chunk_size)
                        if not chunk:
                            break
                        await fo.write(chunk)
                return
        except aiohttp.ClientResponseError as e:
            status = HTTPStatus(e.status)
            message = (
                f"HTTP Status: {status.value}" +
                f" - {status.phrase}" +
                f" - {status.description}\n" +
                f"{url}"
                )
            warnings.warn(message, RuntimeWarning)

        # Sleep
        await asyncio.sleep(5 * 2 ** attempt)

    # Unable to retrieve file
    message = f"Unable to retrieve {url}"
    warnings.warn(message, RuntimeWarning)

async def download_files_awaitable(
        *src_dst: tuple[str | URL, str | Path],
        auto_decompress: bool = True,
        headers: LooseHeaders | None = None,
        limit: int = 100,
        chunk_size: int = 1024,
        overwrite: bool = False,
        ssl_context: ssl.SSLContext | None = None,
        timeout: int = 300,
        retries: int = 3
        ) -> None:
    """
    Asynchronously retrieve multiple files from urls and save to filepaths.
    
    Parameters
    ----------
    *src_dst: tuple[str | Path, str | URL], required
        One or more tuples containing two values. The first value is the 
        source URL from which to retrieve a file, the second value is the
        filepath where the file will be saved.
    auto_decompress: bool, optional, default True
        Automatically decompress responses.
    headers: LooseHeaders | None, default None
        Additional headers to send with each request.
    limit: int, optional, default 100
        Maximum number of simultaneous connections.
    chunk_size: int, optional, default 1024
        Amount of data to write at a time (KB).
    overwrite: bool, optional, default False
        If filepath exists, overwrite if True, else skip download.
    ssl_context: SSLContext, optional
        SSL configuration object. Uses system defaults unless otherwise
        specified.
    timeout: int, optional, default 300
        Maximum number of seconds to wait for a response to return.
    retries: int, optional, default 3
        Number of download attempts in the event of failure.
    
    Returns
    -------
    None
    """
    # SSL
    if ssl_context is None:
        ssl_context = ssl.create_default_context()

    # Retrieve
    connector = aiohttp.TCPConnector(limit=limit)
    async with aiohttp.ClientSession(
        connector=connector,
        headers=headers,
        auto_decompress=auto_decompress
        ) as session:
        await asyncio.gather(*(
            download_file_awaitable(
                url,
                filepath,
                session,
                chunk_size=chunk_size,
                overwrite=overwrite,
                ssl_context=ssl_context,
                timeout=timeout,
                retries=retries
                ) for (url, filepath) in src_dst
        ))

def download_files(
        *src_dst: tuple[str | URL, str | Path],
        auto_decompress: bool = True,
        headers: LooseHeaders | None = None,
        limit: int = 100,
        chunk_size: int = 1024,
        overwrite: bool = False,
        ssl_context: ssl.SSLContext | None = None,
        timeout: int = 300,
        file_validator: Callable[[Path], None] = default_file_validator,
        retries: int = 3
    ) -> None:
    """
    Asynchronously retrieve multiple files from urls and save to filepaths.
    
    Parameters
    ----------
    *src_dst: tuple[str | Path, str | URL], required
        One or more tuples containing two values. The first value is the 
        source URL from which to retrieve a file, the second value is the
        filepath where the file will be saved.
    auto_decompress: bool, optional, default True
        Automatically decompress responses.
    headers: LooseHeaders | None, default None
        Additional headers to send with each request.
    limit: int, optional, default 100
        Maximum number of simultaneous connections.
    chunk_size: int, optional, default 1024
        Amount of data to write at a time (KB).
    overwrite: bool, optional, default False
        If filepath exists, overwrite if True, else skip download.
    ssl_context: SSLContext, optional
        SSL configuration object. Uses system defaults unless otherwise
        specified.
    timeout: int, optional, default 300
        Maximum number of seconds to wait for a response to return.
    file_validator: Callable, optional
        Function used to validate files.
    retries: int, optional, default 3
        Number of download attempts in the event of failure.
    
    Returns
    -------
    None

    Examples
    --------
    >>> # This will download the pandas and numpy homepages and save them to 
    >>> # ./pandas_index.html and ./numpy_index.html
    >>> from downloads import download_files
    >>> download_files(
    ...     ("https://pandas.pydata.org/docs/user_guide/index.html", "pandas_index.html"),
    ...     ("https://numpy.org/doc/stable/index.html", "numpy_index.html")
    ...     )
    """
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    # SSL
    if ssl_context is None:
        ssl_context = ssl.create_default_context()

    # Retrieve
    for attempt in range(retries):
        logger.info("Downloading files, attempt %d", attempt)
        asyncio.run(
            download_files_awaitable(
                *src_dst,
                auto_decompress=auto_decompress,
                headers=headers,
                limit=limit,
                chunk_size=chunk_size,
                overwrite=overwrite,
                ssl_context=ssl_context,
                timeout=timeout,
                retries=retries
                )
            )

        # Validate files
        logger.info("Validating files, this will take some time")
        filepaths = [Path(dst) for _, dst in src_dst]
        validated = 0
        for fp in filepaths:
            logger.info("Validating %s", fp)
            if fp.exists():
                try:
                    file_validator(fp)
                    validated += 1
                except FileValidationError:
                    fp.unlink()
                    break
            else:
                break
        if validated == 0:
            warnings.warn("Unable to retrieve any files", RuntimeWarning)
        if validated == len(filepaths):
            logger.info("All files validated")
            return
        warnings.warn("Unable to retrieve all files, trying again", RuntimeWarning)
        sleep(5 * 2 ** attempt)
    warnings.warn("Unable to validate all files", RuntimeWarning)
