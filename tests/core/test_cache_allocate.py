"""
Test that CacheManager._allocate correctly handles eviction with page_size > 1.
"""

from __future__ import annotations

import minisgl.core as core
import pytest
import torch
from minisgl.scheduler.cache import CacheManager


@pytest.fixture(autouse=True)
def reset_global_ctx():
    """Reset global context before and after each test."""
    old_ctx = core._GLOBAL_CTX
    core._GLOBAL_CTX = None
    yield
    core._GLOBAL_CTX = old_ctx


def _make_cache_manager(num_pages: int, page_size: int) -> CacheManager:
    """Helper to create a CacheManager with radix cache on CPU."""
    page_table = torch.empty((1,))
    ctx = core.Context(page_size=page_size)
    core.set_global_ctx(ctx)
    return CacheManager(num_pages, page_size, page_table, type="radix")


def _insert_evictable(cm: CacheManager, input_ids: torch.Tensor, indices: torch.Tensor):
    """Insert a prefix into the radix cache so it becomes evictable."""
    cm.prefix_cache.insert_prefix(input_ids, indices)


def _assert_all_page_aligned(tensor: torch.Tensor, page_size: int, label: str = ""):
    """Assert every element in tensor is a multiple of page_size."""
    if len(tensor) == 0:
        return
    misaligned = tensor[tensor % page_size != 0]
    assert (
        len(misaligned) == 0
    ), f"{label} contains non-page-aligned values: {misaligned.tolist()}, page_size={page_size}"


def _assert_no_overlap(pages: torch.Tensor, page_size: int):
    """Assert that page-aligned starts, when expanded, produce no overlapping ranges."""
    if len(pages) <= 1:
        return
    expanded = set()
    for p in pages.tolist():
        token_range = set(range(p, p + page_size))
        overlap = expanded & token_range
        assert len(overlap) == 0, f"Overlapping tokens: {overlap}"
        expanded.update(token_range)


class TestAllocateEvictPageAlignment:
    """Tests for _allocate handling eviction with page_size > 1."""

    def test_allocate_after_evict_returns_page_aligned(self):
        """Allocated pages after eviction must be page-aligned."""
        page_size = 4
        num_pages = 4
        cm = _make_cache_manager(num_pages, page_size)

        # Exhaust all free pages
        cm._allocate(num_pages)
        assert len(cm.free_slots) == 0

        # Insert 2 pages worth of data into the cache (evictable)
        input_ids = torch.arange(page_size * 2, dtype=torch.int32)
        # Simulate page table entries: page 0 = [0,1,2,3], page 1 = [4,5,6,7]
        indices = torch.arange(page_size * 2, dtype=torch.int32)
        _insert_evictable(cm, input_ids, indices)

        # Allocate 1 page — triggers eviction
        allocated = cm._allocate(1)
        _assert_all_page_aligned(allocated, page_size, "allocated")
        _assert_all_page_aligned(cm.free_slots, page_size, "_free_slots after evict")

    def test_consecutive_allocations_after_evict_no_overlap(self):
        """Multiple allocations after eviction must not produce overlapping pages."""
        page_size = 4
        num_pages = 4
        cm = _make_cache_manager(num_pages, page_size)

        # Exhaust all free pages
        cm._allocate(num_pages)

        # Insert 2 pages into cache
        input_ids = torch.arange(page_size * 2, dtype=torch.int32)
        indices = torch.arange(page_size * 2, dtype=torch.int32)
        _insert_evictable(cm, input_ids, indices)

        # Allocate 2 pages one by one
        page_a = cm._allocate(1)
        page_b = cm._allocate(1)
        all_pages = torch.cat([page_a, page_b])

        _assert_all_page_aligned(all_pages, page_size, "all_pages")
        _assert_no_overlap(all_pages, page_size)

    def test_free_slots_stay_page_aligned_after_evict(self):
        """_free_slots must remain page-aligned after eviction refills them."""
        page_size = 8
        num_pages = 8
        cm = _make_cache_manager(num_pages, page_size)

        # Exhaust all free pages
        cm._allocate(num_pages)

        # Insert 4 pages worth of data (4 * 8 = 32 tokens)
        n_tokens = page_size * 4
        input_ids = torch.arange(n_tokens, dtype=torch.int32)
        # Indices: page starts at 0, 8, 16, 24
        indices = torch.arange(n_tokens, dtype=torch.int32)
        _insert_evictable(cm, input_ids, indices)

        # Allocate 1 page — evicts and refills _free_slots
        cm._allocate(1)

        # All remaining free slots must be page-aligned
        _assert_all_page_aligned(cm.free_slots, page_size, "_free_slots")

    def test_allocate_exact_pages_needed_from_evict(self):
        """When exactly N pages are needed, eviction must provide at least N pages."""
        page_size = 4
        num_pages = 4
        cm = _make_cache_manager(num_pages, page_size)

        # Exhaust all
        cm._allocate(num_pages)

        # Insert 3 pages into cache
        n_tokens = page_size * 3
        input_ids = torch.arange(n_tokens, dtype=torch.int32)
        indices = torch.arange(n_tokens, dtype=torch.int32)
        _insert_evictable(cm, input_ids, indices)

        # Allocate 2 pages at once — needs eviction of at least 2 pages
        allocated = cm._allocate(2)
        assert len(allocated) == 2
        _assert_all_page_aligned(allocated, page_size, "allocated")
        _assert_no_overlap(allocated, page_size)

    def test_page_to_token_expansion_correct_after_evict(self):
        """_page_to_token on eviction-allocated pages must produce correct consecutive ranges."""
        page_size = 4
        num_pages = 4
        cm = _make_cache_manager(num_pages, page_size)

        cm._allocate(num_pages)

        # Insert 2 pages: tokens [0..7] with indices [0..7]
        input_ids = torch.arange(page_size * 2, dtype=torch.int32)
        indices = torch.arange(page_size * 2, dtype=torch.int32)
        _insert_evictable(cm, input_ids, indices)

        # Allocate 2 pages via eviction
        pages = cm._allocate(2)
        tokens = cm._page_to_token(pages)

        # Each page should expand to page_size consecutive tokens
        assert len(tokens) == 2 * page_size
        for i, page_start in enumerate(pages.tolist()):
            chunk = tokens[i * page_size : (i + 1) * page_size].tolist()
            expected = list(range(page_start, page_start + page_size))
            assert chunk == expected, f"Page {page_start}: got {chunk}, expected {expected}"

    def test_check_integrity_passes_after_evict_cycle(self):
        """CacheManager integrity check should pass after allocate-free-evict cycles."""
        page_size = 4
        num_pages = 8
        cm = _make_cache_manager(num_pages, page_size)

        # Allocate 2 pages (simulating a request using them)
        pages_for_cache = cm._allocate(2)
        # free_slots: 6 pages remaining

        # Insert those 2 pages into radix cache (simulating a finished request)
        # The token indices come from _page_to_token expansion of the allocated pages
        token_indices = cm._page_to_token(pages_for_cache)
        n_tokens = len(token_indices)
        input_ids = torch.arange(n_tokens, dtype=torch.int32)
        _insert_evictable(cm, input_ids, token_indices)
        # Now: free=6, cache=2, total=8

        # This should not raise
        cm.check_integrity()

        # Now exhaust free slots and trigger eviction
        cm._allocate(6)
        assert len(cm.free_slots) == 0

        # Allocate 1 more page — must evict from cache
        allocated = cm._allocate(1)
        _assert_all_page_aligned(allocated, page_size, "allocated after evict")
        _assert_all_page_aligned(cm.free_slots, page_size, "_free_slots after evict")


if __name__ == "__main__":
    pytest.main([__file__])


@pytest.mark.parametrize("page_size", [1, 4, 8])
def test_fragmented_cancel_reallocate_and_evict(page_size):
    cm = _make_cache_manager(8, page_size)
    held = cm._allocate(8)
    # Cancellation releases alternating pages, forcing a fragmented subsequent history.
    cm._free(cm._page_to_token(held[::2]))
    fragmented = cm._allocate(4)
    assert torch.equal(fragmented, held[::2])
    _assert_no_overlap(torch.cat((fragmented, held[1::2])), page_size)
    tokens = cm._page_to_token(fragmented)
    _insert_evictable(cm, torch.arange(len(tokens), dtype=torch.int32), tokens)
    cm._free(cm._page_to_token(held[1::2]))
    cm.check_integrity()
    active = cm._allocate(4)
    reused = cm._allocate(4)  # Evict the fragmented cached history.
    _assert_no_overlap(torch.cat((active, reused)), page_size)
    cm._free(cm._page_to_token(torch.cat((active, reused))))
    cm.check_integrity()
    assert cm.free_slots.unique().numel() == 8


@pytest.mark.parametrize("page_size", [1, 4, 8])
def test_naive_cache_reuse_matches_fresh_attention(page_size, monkeypatch):
    from types import SimpleNamespace

    import minisgl.kvcache.naive_pool as pool_module
    from minisgl.attention.naive import attention
    from minisgl.models.config import ModelConfig
    from minisgl.models.naive_config import NaiveN05FlashConfig

    monkeypatch.setattr(pool_module, "get_tp_info", lambda: SimpleNamespace(size=1))
    c = ModelConfig.from_hf(
        NaiveN05FlashConfig(
            num_hidden_layers=1,
            hybrid_layer_pattern=[0],
            moe_layer_freq=[0],
            num_attention_heads=1,
            num_key_value_heads=1,
        )
    )
    pool = pool_module.NaiveKVCache(c, 8, page_size, torch.float32, torch.device("cpu"))
    cm = _make_cache_manager(8, page_size)
    all_pages = cm._allocate(8)
    cm._free(cm._page_to_token(all_pages[::2]))  # Cancel fragmented history.
    pages = cm._allocate(4)
    locations = cm._page_to_token(pages)
    n = len(locations)
    generator = torch.Generator().manual_seed(51)
    k = torch.randn(n, 1, 192, generator=generator)
    v = torch.randn(n, 1, 128, generator=generator)
    index = torch.randn(n, 128, generator=generator)
    pool.store_kv(k * 99, v * 99, locations, 0)
    pool.store_index(index * 99, locations, 0)
    _insert_evictable(cm, torch.arange(n, dtype=torch.int32), locations)
    cm._free(cm._page_to_token(all_pages[1::2]))
    cm.check_integrity()
    active = cm._allocate(4)
    reused = cm._allocate(4)  # Evict and overwrite old KV/indexer entries.
    reused_locations = cm._page_to_token(reused)
    pool.store_kv(k, v, reused_locations, 0)
    pool.store_index(index, reused_locations, 0)
    actual_k = pool.k_cache(0).flatten(0, 1)[reused_locations.long()]
    actual_v = pool.v_cache(0).flatten(0, 1)[reused_locations.long()]
    torch.testing.assert_close(pool.index_cache(0)[reused_locations.long()], index)
    q = torch.randn(1, 3, 192, generator=generator)
    allowed = torch.ones(3, n, dtype=torch.bool)
    actual = attention(q, actual_k.transpose(0, 1), actual_v.transpose(0, 1), allowed, None, None)
    fresh = attention(q, k.transpose(0, 1), v.transpose(0, 1), allowed, None, None)
    torch.testing.assert_close(actual, fresh)
    cm._free(cm._page_to_token(torch.cat((active, reused))))
    cm.check_integrity()
    assert cm.free_slots.unique().numel() == 8
