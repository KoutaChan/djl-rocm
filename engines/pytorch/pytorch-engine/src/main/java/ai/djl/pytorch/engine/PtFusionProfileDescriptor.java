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
package ai.djl.pytorch.engine;

import ai.djl.engine.fusion.FusionRecipe;
import ai.djl.engine.fusion.FusionShapeProfile;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/** Encodes storage-capacity profiles independently from the stable recipe descriptor. */
final class PtFusionProfileDescriptor {

    static final long MAGIC = 0x444a4c5f46535031L;
    static final long VERSION = 1;
    static final int HEADER_WORDS = 5;

    private PtFusionProfileDescriptor() {}

    static Encoded encode(FusionRecipe recipe, List<FusionShapeProfile> profiles) {
        int dimensionCount = recipe.getDimensions().size();
        int totalWords =
                Math.addExact(HEADER_WORDS, Math.multiplyExact(profiles.size(), dimensionCount));
        ByteBuffer descriptor =
                ByteBuffer.allocateDirect(Math.multiplyExact(totalWords, Long.BYTES))
                        .order(ByteOrder.nativeOrder());
        descriptor.putLong(MAGIC);
        descriptor.putLong(VERSION);
        descriptor.putLong(totalWords);
        descriptor.putLong(profiles.size());
        descriptor.putLong(dimensionCount);

        long[] maximumCapacities = maximumCapacities(recipe);
        List<long[]> capacities = new ArrayList<>(profiles.size());
        Set<CapacityKey> unique = new HashSet<>();
        unique.add(new CapacityKey(maximumCapacities));
        for (FusionShapeProfile profile : profiles) {
            if (profile.getRecipe() != recipe) {
                throw new IllegalArgumentException(
                        "A shape profile belongs to a different fusion recipe.");
            }
            long[] resolved = resolve(recipe, profile);
            if (!unique.add(new CapacityKey(resolved))) {
                throw new IllegalArgumentException(
                        "Fusion storage-capacity profiles must be distinct from each other and "
                                + "from the recipe maximum.");
            }
            capacities.add(resolved);
            for (long capacity : resolved) {
                descriptor.putLong(capacity);
            }
        }
        descriptor.flip();
        return new Encoded(descriptor, new Catalog(maximumCapacities, capacities));
    }

    static long[] maximumCapacities(FusionRecipe recipe) {
        long[] capacities = new long[recipe.getDimensions().size()];
        for (int i = 0; i < capacities.length; ++i) {
            capacities[i] = recipe.getDimensions().get(i).getMaximumExtent();
        }
        return capacities;
    }

    static long[] resolve(FusionRecipe recipe, FusionShapeProfile profile) {
        if (profile.getRecipe() != recipe) {
            throw new IllegalArgumentException(
                    "The requested shape profile belongs to a different fusion recipe.");
        }
        long[] capacities = profile.getCapacities();
        if (capacities.length != recipe.getDimensions().size()) {
            throw new IllegalArgumentException("Invalid Fusion shape profile dimension count.");
        }
        for (int i = 0; i < capacities.length; ++i) {
            if (capacities[i] < 0) {
                capacities[i] = recipe.getDimensions().get(i).getMaximumExtent();
            }
        }
        return capacities;
    }

    static final class Encoded {

        private final ByteBuffer descriptor;
        private final Catalog catalog;

        private Encoded(ByteBuffer descriptor, Catalog catalog) {
            this.descriptor = descriptor;
            this.catalog = catalog;
        }

        ByteBuffer getDescriptor() {
            return descriptor;
        }

        List<long[]> getCapacities() {
            return catalog.capacities;
        }

        Catalog getCatalog() {
            return catalog;
        }
    }

    /** Immutable backend-internal capacities shared by plans, executables, and sessions. */
    static final class Catalog {

        private final long[] maximumCapacities;
        private final List<long[]> capacities;

        private Catalog(long[] maximumCapacities, List<long[]> capacities) {
            this.maximumCapacities = maximumCapacities;
            this.capacities = Collections.unmodifiableList(new ArrayList<>(capacities));
        }

        long[] getMaximumCapacities() {
            return maximumCapacities;
        }

        long[] getCapacities(int profileIndex) {
            return capacities.get(profileIndex);
        }

        int size() {
            return capacities.size();
        }
    }

    private static final class CapacityKey {

        private final long[] capacities;

        private CapacityKey(long[] capacities) {
            this.capacities = capacities;
        }

        @Override
        public boolean equals(Object other) {
            return other instanceof CapacityKey
                    && Arrays.equals(capacities, ((CapacityKey) other).capacities);
        }

        @Override
        public int hashCode() {
            return Arrays.hashCode(capacities);
        }
    }
}
