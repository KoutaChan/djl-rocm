plugins {
    ai.djl.javaProject
    ai.djl.publish
}

val ptVersion: String = when {
    project.hasProperty("pt_version") && project.property("pt_version") != "" ->
        project.property("pt_version").toString()

    else -> libs.versions.pytorch.get()
}

group = "ai.djl.pytorch"
version = ptVersion + '-' + libs.versions.djl.get()
val isRelease = project.hasProperty("release") || project.hasProperty("staging")
if (!isRelease)
    version = ptVersion + "-${libs.versions.djl.get()}-SNAPSHOT"

tasks {
    compileJava { dependsOn(processResources) }

    processResources {
        outputs.dir(buildDirectory / "classes/java/main/jnilib")

        val djlVersion = libs.versions.djl.get()
        val logger = project.logger
        val dir = project.projectDir
        val buildDir = buildDirectory
        val nativeDir = project.parent!!.projectDir / "pytorch-native/jnilib/${djlVersion}/"
        val version = project.version
        val injected = project.objects.newInstance<InjectedOps>()

        doFirst {
            val jnilibDir = dir / "jnilib" / djlVersion
            val outputDir = buildDir / "classes/java/main/jnilib"

            injected.fs.delete {
                delete(outputDir)
            }

            var found = false
            listOf(jnilibDir, nativeDir).forEach { sourceDir ->
                val hasJniLib =
                    sourceDir.exists() &&
                            sourceDir.walkTopDown().any {
                                it.isFile && it.name.contains("djl_torch")
                            }
                if (hasJniLib) {
                    logger.lifecycle("Copying local JNI libraries from $sourceDir")
                    injected.fs.copy {
                        from(sourceDir)
                        into(outputDir)
                    }
                    found = true
                }
            }

            if (!found) {
                throw GradleException(
                    "No local PyTorch JNI libraries found. Build pytorch-native first or place JNI libraries under $jnilibDir."
                )
            }

            // write properties
            val propFile = outputDir / "pytorch.properties"
            propFile.text = "jni_version=$version"
        }
    }

    clean {
        val injected = project.objects.newInstance<InjectedOps>()
        val files = fileTree("$home/.djl.ai/pytorch/") {
            include("**/*djl_torch.*")
        }
        doFirst {
            injected.fs.delete {
                delete("jnilib")
                delete(files)
            }
        }
    }
}

publishing {
    publications {
        named<MavenPublication>("maven") {
            pom {
                name = "DJL Engine Adapter for PyTorch"
                description = "Deep Java Library (DJL) Engine Adapter for PyTorch"
                url = "http://www.djl.ai/engines/pytorch/${project.name}"
            }
        }
    }
}

interface InjectedOps {
    @get:Inject
    val fs: FileSystemOperations
}
