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

package ai.djl.pytorch

import groovy.json.JsonSlurper
import org.gradle.api.DefaultTask
import org.gradle.api.file.DirectoryProperty
import org.gradle.api.file.FileSystemOperations
import org.gradle.api.provider.Property
import org.gradle.api.tasks.Input
import org.gradle.api.tasks.InputDirectory
import org.gradle.api.tasks.OutputDirectory
import org.gradle.api.tasks.PathSensitive
import org.gradle.api.tasks.PathSensitivity
import org.gradle.api.tasks.TaskAction
import org.gradle.process.ExecOperations
import org.gradle.work.DisableCachingByDefault
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.InputStream
import java.nio.channels.FileChannel
import java.nio.file.Files
import java.nio.file.LinkOption.NOFOLLOW_LINKS
import java.nio.file.StandardOpenOption.CREATE
import java.nio.file.StandardOpenOption.WRITE
import java.security.MessageDigest
import java.util.zip.ZipFile
import javax.inject.Inject

/** Builds the pinned hipBLASLt host library and stages its runtime metadata. */
@DisableCachingByDefault(because = "The runtime cache also validates the installed ROCm SDK")
abstract class PrepareHipblaslt @Inject constructor(
    private val execOperations: ExecOperations,
    private val fileOperations: FileSystemOperations
) : DefaultTask() {

    @get:InputDirectory
    @get:PathSensitive(PathSensitivity.RELATIVE)
    abstract val sourceDir: DirectoryProperty

    @get:OutputDirectory
    abstract val outputDir: DirectoryProperty

    @get:Input
    abstract val flavorName: Property<String>

    init {
        outputs.upToDateWhen { false }
    }

    @TaskAction
    fun prepare() {
        check(
            System.getProperty("os.name") == "Linux" &&
                    System.getProperty("os.arch") in listOf("amd64", "x86_64")
        ) {
            "The pinned runtime supports Linux x86_64 only"
        }
        val scripts = sourceDir.get().asFile
        val cache = File(
            System.getenv("HIPBLASLT_CACHE_DIR")
                ?: scripts.parentFile.parentFile.resolve("build/hipblaslt-cache").path
        ).canonicalFile
        cache.mkdirs()
        FileChannel.open(cache.resolve(".prepare.lock").toPath(), CREATE, WRITE).use { channel ->
            channel.lock().use {
                prepare(scripts, cache)
            }
        }
    }

    private fun prepare(scripts: File, cache: File) {
        val sdk = findSdk()
        val profile = readProfile(scripts, sdk)
        val sdkLibrary = sdk.resolve("lib/libhipblaslt.so")
        val dynamic = run("readelf", "-d", sdkLibrary.path, capture = true)
        check(soname(dynamic) == profile.soname) {
            "SDK SONAME differs from pinned ${profile.soname}"
        }
        val dependencies = Regex("\\(NEEDED\\).*\\[(.*?)\\]")
            .findAll(dynamic).map { it.groupValues[1] }.toList()
        val (targets, metadata) = readMetadata(sdk)
        val patchHash = hash(profile.patches.joinToString("") {
            "$it\u0000${sha256(scripts.resolve("patches/$it"))}\n"
        })
        val inputs = sortedMapOf(
            "producer" to "gradle-v1",
            "producer_sha256" to implementationHash(),
            "source_repository" to profile.repository,
            "source_revision" to profile.revision,
            "msgpack_revision" to "8c602e8579c7e7d65d6f9c6703c9699db3fb0488",
            "fmt_revision" to "9cf9f38eded63e5e0fb95cd536ba51be601d7fa2",
            "yaml_revision" to "f7320141120f720aecc4c32be25586e7da9eb978",
            "patch_sha256" to patchHash,
            "profiles_sha256" to sha256(scripts.resolve("profiles.json")),
            "build_script_sha256" to sha256(scripts.resolve("build.sh")),
            "sdk_library_sha256" to sha256(sdkLibrary),
            "kernel_metadata_sha256" to hash(metadata.entries.joinToString("") {
                "${it.key}\u0000${it.value}\n"
            }),
            "gpu_targets" to targets.joinToString(","),
            "sdk_dependencies" to dependencies.sorted().joinToString(","),
            "toolchain" to run(sdk.resolve("bin/amdclang++").path, "--version", capture = true)
                .trim().replace("\n", " | "),
            "cmake" to run("cmake", "--version", capture = true).lineSequence().first(),
            "flavor" to flavorName.get(),
            "classifier" to "linux-x86_64",
            "rocm_version" to profile.version,
            "hipblaslt_version" to profile.hipblasltVersion,
            "soname" to profile.soname,
            "library" to profile.soname
        )
        for (name in listOf("CFLAGS", "CXXFLAGS", "LDFLAGS", "CPLUS_INCLUDE_PATH")) {
            inputs["build_env.$name"] = System.getenv(name) ?: ""
        }
        val fingerprint = hash(inputs.entries.joinToString("") { "${it.key}\u0000${it.value}\n" })
        val artifact = cache.resolve("artifacts/gradle-$fingerprint").canonicalFile
        val output = outputDir.get().asFile.canonicalFile
        check(
            !output.toPath().startsWith(artifact.toPath()) &&
                    !artifact.toPath().startsWith(output.toPath())
        ) {
            "The runtime output must not overlap its cached artifact: $output"
        }
        val manifest = artifact.resolve("hipblaslt.properties")
        val expected = inputs + metadata.mapKeys { "kernel_metadata.${it.key}" }
        if (isComplete(artifact, expected, profile.soname)) {
            logger.lifecycle("Reusing patched hipBLASLt ${flavorName.get()} $fingerprint")
        } else {
            val source = cache.resolve("source/${profile.revision}-$patchHash")
            checkoutSource(source, profile)
            applyPatches(scripts, source, profile)
            val build = cache.resolve("work/gradle-$fingerprint")
            run(
                "bash", scripts.resolve("build.sh").path,
                source.resolve(profile.directory).path, build.path, profile.family,
                environment = mapOf(
                    "ROCM_PATH" to sdk.path,
                    "HIPBLASLT_DEPS_DIR" to build.resolve("deps").path,
                    "GPU_TARGET" to targets.joinToString(";"),
                    "MSGPACK_REVISION" to inputs.getValue("msgpack_revision"),
                    "FMT_REVISION" to inputs.getValue("fmt_revision"),
                    "YAML_REVISION" to inputs.getValue("yaml_revision")
                )
            )
            val library = build.resolve("library/libhipblaslt.so")
            check(soname(run("readelf", "-d", library.path, capture = true)) == profile.soname) {
                "Patched library SONAME differs from SDK ${profile.soname}"
            }
            val missing = readSymbols(sdkLibrary) - readSymbols(library)
            check(missing.isEmpty()) {
                "Patched hipBLASLt is missing SDK public symbols: ${missing.sorted().joinToString(", ")}"
            }
            artifact.mkdirs()
            library.copyTo(artifact.resolve(profile.soname), overwrite = true)
            val licenses = artifact.resolve("licenses")
            fileOperations.delete { delete(licenses) }
            licenses.mkdirs()
            val originals = linkedMapOf(
                "hipblaslt-LICENSE.md" to source.resolve("${profile.directory}/LICENSE.md"),
                "msgpack-COPYING" to build.resolve("deps/msgpack-cxx-6.1.0/source/COPYING"),
                "msgpack-LICENSE_1_0.txt" to build.resolve("deps/msgpack-cxx-6.1.0/source/LICENSE_1_0.txt")
            )
            if (profile.family == "modern") {
                // ROCm 7.1 omits Origami's license file; its MIT notice matches hipBLASLt's.
                originals["origami-LICENSE.md"] = if (profile.version == "7.1.0") {
                    originals.getValue("hipblaslt-LICENSE.md")
                } else {
                    source.resolve("shared/origami/LICENSE.md")
                }
            }
            if (dependencies.any { it.contains("rocroller") }) {
                originals["fmt-LICENSE"] = build.resolve("deps/fmt/source/LICENSE")
                originals["yaml-cpp-LICENSE"] = build.resolve("deps/yaml-cpp/source/LICENSE")
            }
            for ((name, original) in originals) {
                original.copyTo(licenses.resolve(name), overwrite = true)
            }
            val files = listOf(profile.soname) + originals.keys.map { "licenses/$it" }
            val properties = expected + mapOf("sha256" to sha256(artifact.resolve(profile.soname))) +
                    files.associate { "file_sha256.$it" to sha256(artifact.resolve(it)) }
            manifest.writeText(properties.toSortedMap().entries.joinToString("") {
                "${it.key}=${it.value}\n"
            })
        }
        val libraryName = Regex("libhipblaslt\\.so(?:\\.[0-9]+)*")
        fileOperations.delete {
            delete(output.resolve("licenses"))
            delete(output.listFiles { file ->
                file.name.matches(libraryName) &&
                        (Files.isRegularFile(file.toPath(), NOFOLLOW_LINKS) ||
                                Files.isSymbolicLink(file.toPath()))
            }?.toList() ?: emptyList<File>())
        }
        fileOperations.copy {
            from(artifact) { include(profile.soname, "hipblaslt.properties", "licenses/**") }
            into(output)
        }
        logger.lifecycle("Prepared private ${flavorName.get()} hipBLASLt runtime: $output")
    }

    private fun findSdk(): File {
        for (name in listOf("ROCM_PATH", "ROCM_HOME")) {
            System.getenv(name)?.takeIf { it.isNotEmpty() }?.let { return File(it).canonicalFile }
        }
        val path = System.getenv("PATH") ?: ""
        if (path.split(File.pathSeparator).any { File(it, "rocm-sdk").canExecute() }) {
            return File(run("rocm-sdk", "path", "--root", capture = true).trim()).canonicalFile
        }
        val sdk = File("/opt/rocm")
        check(sdk.resolve("bin/hipcc").isFile) {
            "Set ROCM_PATH to the matching ROCm development SDK"
        }
        return sdk.canonicalFile
    }

    private fun readProfile(scripts: File, sdk: File): Profile {
        val version = Regex("^\\d+\\.\\d+\\.\\d+")
            .find(sdk.resolve(".info/version").readText().trim())?.value
        check(version != null) { "Invalid ROCm SDK version: $sdk" }
        check(flavorName.get() == "rocm${version.substringBeforeLast('.')}") {
            "Requested ${flavorName.get()}, but $sdk contains ROCm $version"
        }
        val profiles = JsonSlurper().parse(scripts.resolve("profiles.json")) as Map<*, *>
        val versions = (profiles[flavorName.get()] as? Map<*, *>)?.get("sdk_versions") as? Map<*, *>
        val values = versions?.get(version) as? Map<*, *>
        check(values != null) { "No pinned hipBLASLt profile for ROCm SDK $version" }
        val profile = Profile(
            version,
            values["source_repository"] as String,
            values["source_revision"] as String,
            values["source_dir"] as String,
            values["hipblaslt_version"] as String,
            values["soname"] as String,
            values["build_family"] as String,
            (values["patches"] as List<*>).map { it as String }
        )
        val header = sdk.resolve("include/hipblaslt/hipblaslt-version.h").readText()
        val components = listOf("MAJOR", "MINOR", "PATCH").zip(profile.hipblasltVersion.split('.'))
        for ((suffix, expected) in components) {
            val actual = Regex("#define\\s+HIPBLASLT_VERSION_$suffix\\s+(\\S+)")
                .find(header)?.groupValues?.get(1)
            check(actual == expected) { "SDK $version requires hipBLASLt ${profile.hipblasltVersion}" }
        }
        val revision = Regex("#define\\s+HIPBLASLT_VERSION_TWEAK\\s+(\\S+)")
            .find(header)?.groupValues?.get(1)
        check(
            revision != null && revision.matches(Regex("[0-9a-f]{8,40}")) &&
                    profile.revision.startsWith(revision)
        ) {
            "SDK $version hipBLASLt source differs from ${profile.revision}"
        }
        return profile
    }

    private fun readMetadata(sdk: File): Pair<List<String>, Map<String, String>> {
        val directory = sdk.resolve("lib/hipblaslt/library")
        val pattern = Regex("gfx[0-9a-f]+")
        var files = directory.walkTopDown()
            .filter { it.isFile && (it.name.endsWith(".dat") || it.name.endsWith(".dat.zlib")) }
            .sortedBy { it.relativeTo(directory).invariantSeparatorsPath }
            .associateWith { file ->
                pattern.findAll(file.relativeTo(directory).invariantSeparatorsPath)
                    .map { it.value }.toSet()
            }
        var targets = files.values.flatten().toSortedSet()
        System.getenv("GPU_TARGET")?.takeIf { it.isNotEmpty() }?.let {
            val requested = it.split(';').toSortedSet()
            check(targets.containsAll(requested)) {
                "SDK Tensile metadata does not support requested GPU targets: $requested"
            }
            targets = requested
            files = files.filterValues { architectures ->
                architectures.isEmpty() || architectures.any { target -> target in targets }
            }
        }
        check(targets.isNotEmpty()) { "Missing SDK Tensile metadata: $directory" }
        return targets.toList() to files.keys.associate {
            it.relativeTo(sdk).invariantSeparatorsPath to sha256(it)
        }.toSortedMap()
    }

    private fun checkoutSource(source: File, profile: Profile) {
        if (!source.resolve(".git").isDirectory) {
            run("git", "init", source.path)
        }
        for ((key, value) in mapOf(
            "remote.origin.url" to "https://github.com/${profile.repository}.git",
            "remote.origin.promisor" to "true",
            "remote.origin.partialclonefilter" to "blob:none",
            "extensions.partialClone" to "origin"
        )) {
            run("git", "-C", source.path, "config", key, value)
        }
        run("git", "-C", source.path, "sparse-checkout", "init", "--no-cone")
        run("git", "-C", source.path, "sparse-checkout", "set", "--no-cone", "--stdin",
            input = sourcePatterns(profile).joinToString("\n", postfix = "\n"))
        val head = run("git", "-C", source.path, "rev-parse", "HEAD",
            capture = true, ignoreFailure = true).trim()
        if (head != profile.revision) {
            run("git", "-C", source.path, "fetch", "--depth", "1", "--filter=blob:none",
                "origin", profile.revision)
            run("git", "-C", source.path, "checkout", "--detach", "FETCH_HEAD")
        }
        check(run("git", "-C", source.path, "rev-parse", "HEAD", capture = true).trim() == profile.revision) {
            "Expected ${profile.repository} revision ${profile.revision}"
        }
    }

    private fun sourcePatterns(profile: Profile): List<String> {
        val prefix = if (profile.directory == ".") "" else "${profile.directory}/"
        val directories = mutableListOf("cmake", "library")
        directories += if (profile.family == "legacy") {
            listOf("tensilelite/Tensile/Source", "tensilelite/rocisa")
        } else {
            listOf("device-library", "tensilelite/cmake", "tensilelite/include",
                "tensilelite/src", "tensilelite/rocisa", "tensilelite/tests")
        }
        val patterns = mutableListOf("/*", "!/*/")
        if (prefix.isNotEmpty()) {
            patterns += listOf(
                "/projects/", "!/projects/*/", "/projects/hipblaslt/", "!/projects/hipblaslt/*/"
            )
        }
        patterns += directories.map { "/$prefix$it/" }
        patterns += "!/${prefix}library/src/amd_detail/rocblaslt/src/Tensile/Logic/"
        if (profile.family == "modern") {
            patterns += listOf("/${prefix}tensilelite/*", "!/${prefix}tensilelite/*/")
            patterns += directories.filter { it.startsWith("tensilelite/") }.map { "/$prefix$it/" }
            patterns += listOf("/shared/", "!/shared/*/", "/shared/origami/")
        }
        return patterns
    }

    private fun applyPatches(scripts: File, source: File, profile: Profile) {
        for (name in profile.patches) {
            val command = mutableListOf("git", "-C", source.path, "apply")
            if (name != "0001-bound-tensile-solution-cache.patch" && profile.directory != ".") {
                command += "--directory=${profile.directory}"
            }
            val patch = scripts.resolve("patches/$name").path
            val result = execOperations.exec {
                commandLine(command + listOf("--reverse", "--check", patch))
                isIgnoreExitValue = true
                standardOutput = ByteArrayOutputStream()
                errorOutput = ByteArrayOutputStream()
            }
            if (result.exitValue != 0) {
                run(*(command + listOf("--check", patch)).toTypedArray())
                run(*(command + patch).toTypedArray())
            }
        }
    }

    private fun readSymbols(library: File): Set<String> =
        run("readelf", "--dyn-syms", "--wide", library.path, capture = true).lineSequence()
            .map { it.trim().split(Regex("\\s+")) }
            .filter { it.size >= 8 && it[6] != "UND" }
            .map { it[7].substringBefore('@') }
            .filter {
                it.startsWith("hipblasLt") || it.startsWith("hipblasltExt") || it.contains("hipblaslt_ext")
            }
            .toSet()

    private fun soname(dynamic: String): String =
        checkNotNull(Regex("\\(SONAME\\).*\\[(.*?)\\]").find(dynamic)) {
            "Missing hipBLASLt SONAME"
        }.groupValues[1]

    private fun isComplete(artifact: File, expected: Map<String, String>, soname: String): Boolean {
        val manifest = artifact.resolve("hipblaslt.properties")
        if (!manifest.isFile) {
            return false
        }
        val properties = manifest.readLines().filter { it.contains('=') }
            .associate { it.substringBefore('=') to it.substringAfter('=') }
        val files = properties.filterKeys { it.startsWith("file_sha256.") }
            .mapKeys { it.key.removePrefix("file_sha256.") }
        return expected.all { (key, value) -> properties[key] == value } &&
                soname in files && properties["sha256"] == files[soname] &&
                files.keys.any { it.startsWith("licenses/") } &&
                files.all { (name, digest) ->
                    val file = artifact.resolve(name).canonicalFile
                    file.toPath().startsWith(artifact.canonicalFile.toPath()) &&
                            file.isFile && sha256(file) == digest
                }
    }

    private fun implementationHash(): String {
        val location = File(PrepareHipblaslt::class.java.protectionDomain.codeSource.location.toURI())
        val prefix = "ai/djl/pytorch/PrepareHipblaslt"
        return if (location.isFile) {
            ZipFile(location).use { archive ->
                hash(archive.entries().asSequence()
                    .filter { it.name.startsWith(prefix) && it.name.endsWith(".class") }
                    .sortedBy { it.name }
                    .joinToString("") { "${it.name}\u0000${sha256(archive.getInputStream(it))}\n" })
            }
        } else {
            hash(location.walkTopDown()
                .filter {
                    it.isFile && it.name.endsWith(".class") &&
                            it.relativeTo(location).invariantSeparatorsPath.startsWith(prefix)
                }
                .sortedBy { it.path }
                .joinToString("") {
                    "${it.relativeTo(location).invariantSeparatorsPath}\u0000${sha256(it)}\n"
                })
        }
    }

    private fun sha256(file: File): String = sha256(file.inputStream())

    private fun sha256(stream: InputStream): String {
        val digest = MessageDigest.getInstance("SHA-256")
        stream.buffered().use { input ->
            val buffer = ByteArray(64 * 1024)
            var count = input.read(buffer)
            while (count >= 0) {
                digest.update(buffer, 0, count)
                count = input.read(buffer)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private fun hash(value: String): String = MessageDigest.getInstance("SHA-256")
        .digest(value.toByteArray(Charsets.UTF_8)).joinToString("") { "%02x".format(it) }

    private fun run(
        vararg command: String,
        capture: Boolean = false,
        ignoreFailure: Boolean = false,
        input: String? = null,
        environment: Map<String, String> = emptyMap()
    ): String {
        val output = ByteArrayOutputStream()
        execOperations.exec {
            commandLine(*command)
            environment(environment)
            isIgnoreExitValue = ignoreFailure
            if (capture) {
                standardOutput = output
            }
            if (ignoreFailure) {
                errorOutput = ByteArrayOutputStream()
            }
            if (input != null) {
                standardInput = input.byteInputStream(Charsets.UTF_8)
            }
        }
        return output.toString(Charsets.UTF_8.name())
    }

    private data class Profile(
        val version: String,
        val repository: String,
        val revision: String,
        val directory: String,
        val hipblasltVersion: String,
        val soname: String,
        val family: String,
        val patches: List<String>
    )
}
