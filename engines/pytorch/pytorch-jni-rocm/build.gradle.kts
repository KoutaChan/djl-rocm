plugins {
    ai.djl.javaProject
    ai.djl.publish
}

val ptVersion: String = when {
    project.hasProperty("pt_version") && project.property("pt_version") != "" ->
        project.property("pt_version").toString()

    else -> libs.versions.pytorch.get()
}

val rocmFlavor: String = when {
    project.hasProperty("rocm_flavor") && project.property("rocm_flavor") != "" ->
        project.property("rocm_flavor").toString()

    else -> "rocm6.3"
}

group = "ai.djl.pytorch"
val isRelease = project.hasProperty("release") || project.hasProperty("staging")
version = ptVersion + '-' + libs.versions.djl.get() + if (isRelease) "" else "-SNAPSHOT"

val stageJniLib = tasks.register("stageJniLib") {
    val djlVersion = libs.versions.djl.get()
    val logger = project.logger
    val nativeDir = project.parent!!.projectDir / "pytorch-native"
    val flavor = rocmFlavor
    val publishedVersion = project.version
    val stageDir = layout.buildDirectory.dir("jnilib-stage")
    val injected = project.objects.newInstance<InjectedOps>()

    outputs.dir(stageDir)
    outputs.upToDateWhen { false }

    doLast {
        val classifier = "linux-x86_64"
        val entry = "$classifier/$flavor/libdjl_torch.so"
        val targetFile = stageDir.get().asFile / entry
        targetFile.parentFile.mkdirs()

        // Prefer a freshly built .so, fall back to the jnilib tree populated
        // by pytorch-native:compileJNI.
        val freshFile = nativeDir / "build/libdjl_torch.so"
        val cachedFile = nativeDir / "jnilib/${djlVersion}/$entry"
        val source = when {
            freshFile.exists() -> freshFile
            cachedFile.exists() -> cachedFile
            else -> throw GradleException(
                "libdjl_torch.so not found. Build it first:\n" +
                        "  cd ${nativeDir} && ./gradlew compileJNI -Pcuda=$flavor"
            )
        }
        logger.lifecycle("Bundling $source into jnilib/$entry")
        injected.fs.copy {
            from(source)
            into(targetFile.parent)
        }

        (stageDir.get().asFile / "pytorch.properties").text =
                "jni_version=$publishedVersion\n"
    }
}

tasks {
    jar {
        dependsOn(stageJniLib)
        from(stageJniLib.map { it.outputs.files }) {
            into("jnilib")
        }
        archiveBaseName = "pytorch-jni-$rocmFlavor"
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
            artifactId = "pytorch-jni-$rocmFlavor"
            pom {
                name = "DJL Engine Adapter for PyTorch (ROCm JNI)"
                description =
                        "DJL PyTorch JNI (libdjl_torch.so) built against ROCm libtorch." +
                                " The libtorch binaries themselves are fetched at runtime" +
                                " by LibUtils from download.pytorch.org."
                url = "http://www.djl.ai/engines/pytorch/${project.name}"
            }
        }
    }
}

interface InjectedOps {
    @get:Inject
    val fs: FileSystemOperations
}
