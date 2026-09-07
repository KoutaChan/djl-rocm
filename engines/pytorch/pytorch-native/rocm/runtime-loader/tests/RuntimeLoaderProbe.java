/*
 * Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package ai.djl.pytorch.jni;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/** CPU fixture proving that only the patched ELF dependency is loaded through the overlay. */
public final class RuntimeLoaderProbe {

    private RuntimeLoaderProbe() {}

    /**
     * Loads the fixture libraries and checks their resolved dependencies.
     *
     * @param args the loading mode, fixture directory, patched library and overlay library
     * @throws Exception if the fixture cannot be loaded or inspected
     */
    public static void main(String[] args) throws Exception {
        boolean canonical = "canonical".equals(args[0]);
        if (canonical) {
            System.load(args[2]);
            System.load(args[3]);
        } else {
            RocmLibraryLoader.initialize(Path.of(args[1]));
            RocmLibraryLoader.load(args[2]);
            RocmLibraryLoader.load(args[3]);
            try {
                RocmLibraryLoader.load(args[1] + "/missing.so");
                throw new AssertionError("Missing library was accepted");
            } catch (UnsatisfiedLinkError expected) {
                if (!expected.getMessage().contains("missing.so")) {
                    throw expected;
                }
            }
        }
        List<String> libraries =
                Files.readAllLines(Path.of("/proc/self/maps")).stream()
                        .filter(line -> line.contains("/libdjl_loader_sample.so"))
                        .map(line -> line.substring(line.indexOf('/')))
                        .distinct()
                        .toList();
        boolean original = libraries.stream().anyMatch(path -> path.contains("/original/"));
        boolean patched = libraries.stream().anyMatch(path -> path.contains("/patched/"));
        if (!patched || original != canonical || libraries.size() != (canonical ? 2 : 1)) {
            throw new AssertionError(args[0] + " loaded unexpected dependencies: " + libraries);
        }
        System.out.println("PASS " + args[0] + ": " + libraries);
    }
}
