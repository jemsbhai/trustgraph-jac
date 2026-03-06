"""Tests for tools/search.py — the sync search bridge for Jac walkers."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.search import _run, _to_dicts, bing_search, brave_search, multi_search, web_search


# ── Helper fixtures ───────────────────────────────────────────────────────────


@dataclass
class FakeSearchResult:
    title: str
    url: str
    content: str
    score: float


@pytest.fixture
def fake_results():
    return [
        FakeSearchResult(
            title="Result 1",
            url="https://example.com/1",
            content="Content 1",
            score=0.9,
        ),
        FakeSearchResult(
            title="Result 2",
            url="https://example.com/2",
            content="Content 2",
            score=0.7,
        ),
    ]


# ── _run ──────────────────────────────────────────────────────────────────────


class TestRun:
    """Tests for the _run async-to-sync helper."""

    def test_nest_asyncio_path_when_loop_running(self):
        """When an event loop is already running, _run should use nest_asyncio."""
        mock_loop = MagicMock()
        mock_loop.run_until_complete = MagicMock(return_value="result")

        async def fake_coro():
            return "result"

        coro = fake_coro()

        with (
            patch("tools.search.asyncio.get_running_loop", return_value=mock_loop),
            patch.dict("sys.modules", {"nest_asyncio": MagicMock()}),
        ):
            result = _run(coro)

        mock_loop.run_until_complete.assert_called_once()
        assert result == "result"
        # Close the coroutine that was passed to the mock (never truly awaited)
        passed_coro = mock_loop.run_until_complete.call_args[0][0]
        passed_coro.close()


# ── _to_dicts ─────────────────────────────────────────────────────────────────


class TestToDicts:
    """Tests for the result-to-dict converter."""

    def test_converts_objects_to_dicts(self, fake_results):
        dicts = _to_dicts(fake_results)
        assert len(dicts) == 2
        assert dicts[0] == {
            "title": "Result 1",
            "url": "https://example.com/1",
            "content": "Content 1",
            "score": 0.9,
        }

    def test_empty_list(self):
        assert _to_dicts([]) == []

    def test_preserves_all_fields(self, fake_results):
        dicts = _to_dicts(fake_results)
        for d in dicts:
            assert set(d.keys()) == {"title", "url", "content", "score"}


# ── web_search ────────────────────────────────────────────────────────────────


class TestWebSearch:
    """Tests for web_search() with mocked TavilySearch."""

    @patch("trustandverify.search.tavily.TavilySearch")
    def test_returns_dicts(self, MockTavily, fake_results):
        instance = MockTavily.return_value
        instance.search = AsyncMock(return_value=fake_results)
        result = web_search("test query", max_results=2)
        assert len(result) == 2
        assert result[0]["title"] == "Result 1"
        assert result[1]["url"] == "https://example.com/2"

    @patch("trustandverify.search.tavily.TavilySearch")
    def test_passes_query_and_max_results(self, MockTavily, fake_results):
        instance = MockTavily.return_value
        instance.search = AsyncMock(return_value=fake_results)
        web_search("my query", max_results=10)
        instance.search.assert_called_once_with("my query", 10)

    @patch("trustandverify.search.tavily.TavilySearch")
    def test_empty_results(self, MockTavily):
        instance = MockTavily.return_value
        instance.search = AsyncMock(return_value=[])
        result = web_search("nothing here")
        assert result == []


# ── brave_search ──────────────────────────────────────────────────────────────


class TestBraveSearch:
    """Tests for brave_search() with mocked BraveSearch."""

    @patch("trustandverify.search.brave.BraveSearch")
    def test_returns_dicts(self, MockBrave, fake_results):
        instance = MockBrave.return_value
        instance.search = AsyncMock(return_value=fake_results)
        result = brave_search("test query", max_results=3)
        assert len(result) == 2
        assert result[0]["title"] == "Result 1"

    @patch("trustandverify.search.brave.BraveSearch")
    def test_passes_query_and_max_results(self, MockBrave, fake_results):
        instance = MockBrave.return_value
        instance.search = AsyncMock(return_value=fake_results)
        brave_search("brave query", max_results=7)
        instance.search.assert_called_once_with("brave query", 7)


# ── bing_search ───────────────────────────────────────────────────────────────


class TestBingSearch:
    """Tests for bing_search() with mocked BingSearch."""

    @patch("trustandverify.search.bing.BingSearch")
    def test_returns_dicts(self, MockBing, fake_results):
        instance = MockBing.return_value
        instance.search = AsyncMock(return_value=fake_results)
        result = bing_search("test query", max_results=3)
        assert len(result) == 2
        assert result[1]["content"] == "Content 2"

    @patch("trustandverify.search.bing.BingSearch")
    def test_passes_query_and_max_results(self, MockBing, fake_results):
        instance = MockBing.return_value
        instance.search = AsyncMock(return_value=fake_results)
        bing_search("bing query", max_results=4)
        instance.search.assert_called_once_with("bing query", 4)


# ── multi_search ──────────────────────────────────────────────────────────────


class TestMultiSearch:
    """Tests for multi_search() with mocked backends."""

    @patch("trustandverify.search.bing.BingSearch")
    @patch("trustandverify.search.brave.BraveSearch")
    @patch("trustandverify.search.tavily.TavilySearch")
    @patch("trustandverify.search.multi.MultiSearch")
    def test_falls_back_to_web_search_when_single_backend(
        self, MockMulti, MockTavily, MockBrave, MockBing, fake_results
    ):
        """If only one backend is available, should use web_search instead."""
        MockTavily.return_value.is_available.return_value = True
        MockTavily.return_value.search = AsyncMock(return_value=fake_results)
        MockBrave.return_value.is_available.return_value = False
        MockBing.return_value.is_available.return_value = False

        result = multi_search("test query", max_results=5)
        MockMulti.assert_not_called()
        assert len(result) == 2

    @patch("trustandverify.search.bing.BingSearch")
    @patch("trustandverify.search.brave.BraveSearch")
    @patch("trustandverify.search.tavily.TavilySearch")
    @patch("trustandverify.search.multi.MultiSearch")
    def test_uses_multisearch_when_multiple_backends(
        self, MockMulti, MockTavily, MockBrave, MockBing, fake_results
    ):
        """If multiple backends available, should use MultiSearch."""
        MockTavily.return_value.is_available.return_value = True
        MockBrave.return_value.is_available.return_value = True
        MockBing.return_value.is_available.return_value = False

        multi_instance = MockMulti.return_value
        multi_instance.search = AsyncMock(return_value=fake_results)

        result = multi_search("test query", max_results=5)
        MockMulti.assert_called_once()
        assert len(result) == 2

    @patch("trustandverify.search.bing.BingSearch")
    @patch("trustandverify.search.brave.BraveSearch")
    @patch("trustandverify.search.tavily.TavilySearch")
    @patch("trustandverify.search.multi.MultiSearch")
    def test_falls_back_when_no_backends(
        self, MockMulti, MockTavily, MockBrave, MockBing
    ):
        """If no backends available, falls back to web_search."""
        MockTavily.return_value.is_available.return_value = False
        MockTavily.return_value.search = AsyncMock(return_value=[])
        MockBrave.return_value.is_available.return_value = False
        MockBing.return_value.is_available.return_value = False

        result = multi_search("test query", max_results=5)
        MockMulti.assert_not_called()
