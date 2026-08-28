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
package ai.djl.training;

import ai.djl.ndarray.types.DataType;
import ai.djl.training.loss.Loss;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class DefaultTrainingConfigAutocastTest {

    @Test
    public void bfloat16AutocastDoesNotInstallScaler() {
        DefaultTrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss()).optAutocast(DataType.BFLOAT16);

        Assert.assertEquals(config.getAutocastDataType().orElse(null), DataType.BFLOAT16);
        Assert.assertTrue(config.isAutocastCacheEnabled());
        Assert.assertFalse(config.getGradScaler().isPresent());
    }

    @Test
    public void float16AutocastInstallsScaler() {
        DefaultTrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss()).optAutocast(DataType.FLOAT16, false);
        GradScaler scaler = config.getGradScaler().orElseThrow(AssertionError::new);
        config.optAutocast(DataType.FLOAT16, true);

        Assert.assertEquals(config.getAutocastDataType().orElse(null), DataType.FLOAT16);
        Assert.assertTrue(config.isAutocastCacheEnabled());
        Assert.assertTrue(config.getGradScaler().isPresent());
        Assert.assertEquals(config.getGradScaler().get().getScale(), 65536f);
        Assert.assertSame(config.getGradScaler().orElse(null), scaler);
    }

    @Test
    public void customScalerIsPreserved() {
        GradScaler scaler = GradScaler.builder().optInitialScale(128f).build();
        DefaultTrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .optGradScaler(scaler)
                        .optAutocast(DataType.FLOAT16);

        Assert.assertSame(config.getGradScaler().orElse(null), scaler);
    }

    @Test
    public void changingToBfloat16OnlyRemovesAutomaticScaler() {
        DefaultTrainingConfig automatic =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .optAutocast(DataType.FLOAT16)
                        .optAutocast(DataType.BFLOAT16);
        Assert.assertFalse(automatic.getGradScaler().isPresent());

        GradScaler scaler = GradScaler.builder().optInitialScale(128f).build();
        DefaultTrainingConfig custom =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .optGradScaler(scaler)
                        .optAutocast(DataType.FLOAT16)
                        .optAutocast(DataType.BFLOAT16);
        Assert.assertSame(custom.getGradScaler().orElse(null), scaler);
    }

    @Test
    public void scalerStateCanBeRestored() {
        GradScaler source = GradScaler.builder().optInitialScale(128f).optGrowthInterval(4).build();
        GradScaler restored = GradScaler.builder().optInitialScale(2f).optGrowthInterval(4).build();

        restored.loadState(new GradScaler.State(64f, 3));
        GradScaler.State state = restored.getState();
        Assert.assertEquals(state.getScale(), 64f);
        Assert.assertEquals(state.getGrowthCount(), 3);

        source.loadState(state);
        Assert.assertEquals(source.getScale(), 64f);
        Assert.assertEquals(source.getGrowthCount(), 3);
        Assert.assertFalse(source.isLastStepSkipped());
    }

    @Test
    public void scalerStateFilePreservesConfigurationAndProgress() throws IOException {
        Path path = Files.createTempFile("grad-scaler-state", ".bin");
        try {
            GradScaler source =
                    GradScaler.builder()
                            .optInitialScale(128f)
                            .optGrowthFactor(4f)
                            .optBackoffFactor(0.25f)
                            .optGrowthInterval(8)
                            .build();
            source.loadState(new GradScaler.State(64f, 3));
            source.saveState(path);

            GradScaler restored =
                    GradScaler.builder()
                            .optInitialScale(2f)
                            .optGrowthFactor(4f)
                            .optBackoffFactor(0.25f)
                            .optGrowthInterval(8)
                            .build();
            restored.loadState(path);
            Assert.assertEquals(restored.getScale(), 64f);
            Assert.assertEquals(restored.getGrowthCount(), 3);

            GradScaler incompatible = GradScaler.builder().optGrowthInterval(4).build();
            Assert.assertThrows(IOException.class, () -> incompatible.loadState(path));
        } finally {
            Files.deleteIfExists(path);
        }
    }

    @Test
    public void rejectsUnsupportedAutocastDataType() {
        DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.l2Loss());
        Assert.assertThrows(
                IllegalArgumentException.class, () -> config.optAutocast(DataType.FLOAT32));
    }

    @Test
    public void validatesScalerConfiguration() {
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> GradScaler.builder().optInitialScale(0f).build());
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> GradScaler.builder().optGrowthFactor(1f).build());
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> GradScaler.builder().optBackoffFactor(1f).build());
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> GradScaler.builder().optGrowthInterval(0).build());
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> GradScaler.builder().optInitialScale(Float.NaN).build());
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> GradScaler.builder().optGrowthFactor(Float.POSITIVE_INFINITY).build());
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> GradScaler.builder().optBackoffFactor(Float.NaN).build());
    }
}
