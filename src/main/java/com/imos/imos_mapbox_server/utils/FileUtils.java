package com.imos.imos_mapbox_server.utils;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

public class FileUtils {
    /**
     * Checks whether a target subdirectory exists under a parent directory and is not empty.
     *
     * <p>The method searches for immediate subdirectories within the parent directory whose name contains
     * the specified target name. It then checks if the matching subdirectory contains at least one file or subdirectory.</p>
     *
     * @param parentDirectory the path to the parent directory
     * @param targetDirectoryName the partial name of the target subdirectory to search for
     * @return {@code true} if a matching non-empty subdirectory exists; {@code false} otherwise
     * @throws IOException if an I/O error occurs accessing the file system
     */
    public static boolean directoryExists(String parentDirectory, String targetDirectoryName) throws IOException {
        Path parentPath = Paths.get(parentDirectory);

        if (!Files.exists(parentPath) || !Files.isDirectory(parentPath)) {
            return false;
        }

        try (Stream<Path> paths = Files.walk(parentPath, 1)) {
            return paths
                    .filter(Files::isDirectory)
                    .filter(path -> path.getFileName().toString().contains(targetDirectoryName))
                    .anyMatch(path -> {
                        try (Stream<Path> subFiles = Files.list(path)) {
                            return subFiles.findAny().isPresent();
                        } catch (IOException e) {
                            return false;
                        }
                    });
        }
    }

    /**
     * Finds all target directory names that are missing or empty under the specified parent directory.
     *
     * <p>The method checks each target name to see if a matching non-empty subdirectory exists.
     * If a target is missing or empty, it is added to the result list.</p>
     *
     * <p>If any IOException occurs during scanning, the method will conservatively return the full list of targets
     * assuming that validation failed.</p>
     *
     * @param parentDirectory the path to the parent directory
     * @param targetDirectoryList the list of partial names for target directories to check
     * @return a {@link List} of target names that were missing or empty
     */
    public static List<String> findMissingDirectories(String parentDirectory, List<String> targetDirectoryList)  {
        List<String> missingFiles = new ArrayList<>();

        for(String fileName : targetDirectoryList) {
            try {
                if(!directoryExists(parentDirectory, fileName)) {
                    missingFiles.add(fileName);
                }
            } catch (IOException e) {
                return targetDirectoryList;
            }
        }

        return missingFiles;
    }

}
