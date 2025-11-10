"""Test thread-local RNG functionality.

Verifies that:
1. Each thread gets independent RNG
2. Reseed is deterministic per thread
3. Two threads with same seed produce identical sequences
4. Thread RNG doesn't affect other threads
"""

import threading
import varwg
import numpy as np


class TestThreadLocalRNG:
    """Test thread-local RNG behavior."""

    def test_get_rng_returns_generator(self):
        """get_rng() returns a numpy Generator instance."""
        rng = varwg.get_rng()
        assert isinstance(rng, np.random.Generator)

    def test_get_rng_same_in_thread(self):
        """Multiple get_rng() calls in same thread return same object."""
        rng1 = varwg.get_rng()
        rng2 = varwg.get_rng()
        assert rng1 is rng2

    def test_reseed_deterministic(self):
        """Reseeding with same seed produces same sequence."""
        varwg.reseed(42)
        seq1 = [varwg.get_rng().normal() for _ in range(5)]

        varwg.reseed(42)
        seq2 = [varwg.get_rng().normal() for _ in range(5)]

        np.testing.assert_array_equal(seq1, seq2)

    def test_different_seeds_different_sequences(self):
        """Different seeds produce different sequences."""
        varwg.reseed(42)
        seq1 = [varwg.get_rng().normal() for _ in range(5)]

        varwg.reseed(100)
        seq2 = [varwg.get_rng().normal() for _ in range(5)]

        # Extremely unlikely to be equal (would be floating point miracle)
        assert not np.allclose(seq1, seq2)

    def test_thread_isolation(self):
        """Each thread gets independent RNG."""
        results = {}

        def worker(thread_id, seed):
            varwg.reseed(seed)
            values = [varwg.get_rng().normal() for _ in range(3)]
            results[thread_id] = values

        # Start two threads with same seed
        t1 = threading.Thread(target=worker, args=(1, 42))
        t2 = threading.Thread(target=worker, args=(2, 42))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Both threads should produce identical sequences (same seed)
        np.testing.assert_array_equal(results[1], results[2])

    def test_thread_no_interference(self):
        """One thread's RNG changes don't affect others."""
        results = {}

        def worker(thread_id, seed):
            varwg.reseed(seed)
            # Generate 3 values
            values = [varwg.get_rng().normal() for _ in range(3)]
            results[thread_id] = values

        # Thread 1: seed 42
        # Thread 2: seed 100
        t1 = threading.Thread(target=worker, args=(1, 42))
        t2 = threading.Thread(target=worker, args=(2, 100))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Sequences should be different
        assert not np.allclose(results[1], results[2])
