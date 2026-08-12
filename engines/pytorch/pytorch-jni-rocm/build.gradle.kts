import java.security.MessageDigest

plugins {
    ai.djl.javaProject
    ai.djl.publish
}

val ptVersion: String = when {
    project.hasProperty("pt_version") && project.property("pt_version") != "" ->
        project.property("pt_version").toString()

    else -> libs.versions.pytorch.get()
}

// `flavor` covers everything published under download.pytorch.org/libtorch/:
// cpu, cu121, cu124, cu128, rocm6.4, rocm7.0, rocm7.1, rocm7.2, ... The
// legacy `rocm_flavor` property is kept as a fallback for callers still
// passing the ROCm-specific name.
val flavor: String = when {
    project.hasProperty("flavor") && project.property("flavor") != "" ->
        project.property("flavor").toString()

    project.hasProperty("rocm_flavor") && project.property("rocm_flavor") != "" ->
        project.property("rocm_flavor").toString()

    else -> "rocm6.4"
}

// OS+arch classifier matching DJL's native-jar conventions.
val classifier: String = when {
    project.hasProperty("classifier") && project.property("classifier") != "" ->
        project.property("classifier").toString()

    else -> "linux-x86_64"
}

// pytorch-native's build output file name depends on the target OS.
val jniLibFileName: String = when {
    classifier.startsWith("win") -> "djl_torch.dll"
    classifier.startsWith("osx") -> "libdjl_torch.dylib"
    else -> "libdjl_torch.so"
}

// GitHub Packages Maven registry mangles maven-metadata.xml whenever a
// Maven classifier containing dashes / digits is uploaded (linux-x86_64
// -> classifier="linux-x", extension="6_64.jar"; see
// github.com/orgs/community/discussions/49682). The corrupted metadata
// makes snapshot resolution fail client-side even though the jar itself
// is served correctly. Fold the OS classifier into the artifactId and
// drop the Maven classifier entirely so every publish lands in its own
// artifactId and never needs metadata classifier parsing. The jar's
// internal layout (jnilib/<classifier>/<flavor>/<lib>) is unchanged so
// DJL's runtime LibUtils still finds the native untouched.
val artifactSlug = "pytorch-jni-$flavor-$classifier"

group = "ai.djl.pytorch"
val isRelease = project.hasProperty("release") || project.hasProperty("staging")
version = ptVersion + '-' + libs.versions.djl.get() + if (isRelease) "" else "-SNAPSHOT"

val stageJniLib = tasks.register("stageJniLib") {
    val djlVersion = libs.versions.djl.get()
    val logger = project.logger
    val nativeDir = project.parent!!.projectDir / "pytorch-native"
    val flavorName = flavor
    val classifierName = classifier
    val libName = jniLibFileName
    val publishedVersion = project.version
    val stageDir = layout.buildDirectory.dir("jnilib-stage")
    val injected = project.objects.newInstance<InjectedOps>()

    outputs.dir(stageDir)
    outputs.upToDateWhen { false }

    doLast {
        val entry = "$classifierName/$flavorName/$libName"
        val targetFile = stageDir.get().asFile / entry
        targetFile.parentFile.mkdirs()

        // Prefer a freshly built library, fall back to the jnilib tree
        // populated by pytorch-native:compileJNI (CI mode).
        val freshFile = nativeDir / "build/$libName"
        val cachedFile = nativeDir / "jnilib/${djlVersion}/$entry"
        val source = when {
            freshFile.exists() -> freshFile
            cachedFile.exists() -> cachedFile
            else -> throw GradleException(
                "$libName not found. Build it first:\n" +
                        "  cd $nativeDir && ./build.sh <pt_version> $flavorName <cxx11|precxx11> <arch>"
            )
        }
        logger.lifecycle("Bundling $source into jnilib/$entry")
        injected.fs.copy {
            from(source)
            into(targetFile.parent)
        }

        val digest = MessageDigest.getInstance("SHA-256")
        targetFile.inputStream().buffered().use { input ->
            val buffer = ByteArray(64 * 1024)
            while (true) {
                val read = input.read(buffer)
                if (read < 0) {
                    break
                }
                digest.update(buffer, 0, read)
            }
        }
        val contentHash = digest.digest().joinToString("") { "%02x".format(it) }

        (stageDir.get().asFile / "pytorch.properties").text =
                "jni_version=$publishedVersion\n" +
                        "jni_cache_key=$djlVersion-$contentHash\n"
    }
}

tasks {
    jar {
        dependsOn(stageJniLib)
        from(stageJniLib.map { it.outputs.files }) {
            into("jnilib")
        }
        archiveBaseName = artifactSlug
        archiveClassifier = ""
    }

    clean {
        val injected = project.objects.newInstance<InjectedOps>()
        doFirst {
            injected.fs.delete { delete("jnilib") }
        }
    }
}

publishing {
    publications {
        named<MavenPublication>("maven") {
            artifactId = artifactSlug
            pom {
                name = "DJL PyTorch JNI ($flavor / $classifier)"
                description =
                        "DJL PyTorch JNI ($jniLibFileName) for the $flavor libtorch flavor on" +
                                " $classifier. The libtorch binaries themselves are fetched at" +
                                " runtime by LibUtils from download.pytorch.org."
                url = "http://www.djl.ai/engines/pytorch/${project.name}"
            }
        }
    }
}

interface InjectedOps {
    @get:Inject
    val fs: FileSystemOperations
}
