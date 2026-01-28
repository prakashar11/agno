import json
import os
from pathlib import Path
from typing import Any, List, Optional

from agno.tools import Toolkit
from agno.utils.log import log_debug, log_error, log_info


class FileTools(Toolkit):
    def __init__(
        self,
        base_dir: Optional[Path] = None,
        save_files: bool = True,
        read_files: bool = True,
        list_files: bool = True,
        search_files: bool = True,
        **kwargs,
    ):
        self.base_dir: Path = base_dir or Path.cwd()

        tools: List[Any] = []
        if save_files:
            tools.append(self.save_file)
        if read_files:
            tools.append(self.read_file)
        if list_files:
            tools.append(self.list_files)
        if search_files:
            tools.append(self.search_files)

        super().__init__(name="file_tools", tools=tools, **kwargs)

    def _expand_path(self, file_name: str) -> Path:
        """
        Expand file path, handling tilde (~) for home directory and absolute paths.
        
        :param file_name: The file name/path (may contain ~ or be absolute)
        :return: Expanded Path object
        """
        # Expand tilde to home directory
        expanded = os.path.expanduser(file_name)
        file_path = Path(expanded)
        
        # If it's an absolute path, use it directly
        if file_path.is_absolute():
            return file_path
        
        # Otherwise, make it relative to base_dir
        return self.base_dir.joinpath(file_path)

    def save_file(self, contents: str, file_name: str, overwrite: bool = True) -> str:
        """Saves the contents to a file called `file_name` and returns the file name if successful.
        
        Supports tilde (~) expansion for home directory paths, e.g., "~/.config/file.txt"

        :param contents: The contents to save.
        :param file_name: The name of the file to save to (supports ~ for home directory).
        :param overwrite: Overwrite the file if it already exists.
        :return: The file name if successful, otherwise returns an error message.
        """
        try:
            file_path = self._expand_path(file_name)
            log_debug(f"Saving contents to {file_path}")
            if not file_path.parent.exists():
                file_path.parent.mkdir(parents=True, exist_ok=True)
            if file_path.exists() and not overwrite:
                return f"File {file_name} already exists"
            file_path.write_text(contents)
            log_info(f"Saved: {file_path}")
            return str(file_name)
        except Exception as e:
            log_error(f"Error saving to file: {e}")
            return f"Error saving to file: {e}"

    def read_file(self, file_name: str) -> str:
        """Reads the contents of the file `file_name` and returns the contents if successful.
        
        Supports tilde (~) expansion for home directory paths, e.g., "~/.clawdbot/clawdbot.json"

        :param file_name: The name of the file to read (supports ~ for home directory).
        :return: The contents of the file if successful, otherwise returns an error message.
        """
        try:
            log_info(f"Reading file: {file_name}")
            file_path = self._expand_path(file_name)
            contents = file_path.read_text(encoding="utf-8")
            return str(contents)
        except Exception as e:
            log_error(f"Error reading file: {e}")
            return f"Error reading file: {e}"

    def list_files(self) -> str:
        """Returns a list of files in the base directory

        :return: The contents of the file if successful, otherwise returns an error message.
        """
        try:
            log_info(f"Reading files in : {self.base_dir}")
            return json.dumps([str(file_path) for file_path in self.base_dir.iterdir()], indent=4)
        except Exception as e:
            log_error(f"Error reading files: {e}")
            return f"Error reading files: {e}"

    def search_files(self, pattern: str) -> str:
        """Searches for files in the base directory that match the pattern.
        
        Supports tilde (~) expansion in patterns, e.g., "~/.config/*.json"

        :param pattern: The pattern to search for, e.g. "*.txt", "file*.csv", "**/*.py", "~/.config/*.json".
        :return: JSON formatted list of matching file paths, or error message.
        """
        try:
            if not pattern or not pattern.strip():
                return "Error: Pattern cannot be empty"

            # Expand tilde in pattern
            expanded_pattern = os.path.expanduser(pattern)
            pattern_path = Path(expanded_pattern)
            
            # If pattern is absolute, search from that directory
            if pattern_path.is_absolute():
                search_dir = pattern_path.parent
                search_pattern = pattern_path.name
            else:
                search_dir = self.base_dir
                search_pattern = expanded_pattern

            log_debug(f"Searching files in {search_dir} with pattern {search_pattern}")
            matching_files = list(search_dir.glob(search_pattern))

            file_paths = [str(file_path) for file_path in matching_files]

            result = {
                "pattern": pattern,
                "base_directory": str(search_dir),
                "matches_found": len(file_paths),
                "files": file_paths,
            }
            log_debug(f"Found {len(file_paths)} files matching pattern {pattern}")
            return json.dumps(result, indent=2)

        except Exception as e:
            error_msg = f"Error searching files with pattern '{pattern}': {e}"
            log_error(error_msg)
            return error_msg
