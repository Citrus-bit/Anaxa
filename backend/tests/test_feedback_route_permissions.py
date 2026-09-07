from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from app.gateway.authz import AuthContext, Permissions
from app.gateway.routers import feedback, threads


def _write_only_request(feedback_repo: MagicMock) -> SimpleNamespace:
    user = SimpleNamespace(id="user-1")
    thread_store = SimpleNamespace(check_access=AsyncMock(return_value=True))
    return SimpleNamespace(
        state=SimpleNamespace(
            auth=AuthContext(user=user, permissions=[Permissions.THREADS_WRITE]),
        ),
        app=SimpleNamespace(
            state=SimpleNamespace(
                feedback_repo=feedback_repo,
                thread_store=thread_store,
            ),
        ),
    )


@pytest.mark.asyncio
async def test_write_permission_can_clear_own_run_feedback(monkeypatch) -> None:
    feedback_repo = MagicMock()
    feedback_repo.delete_by_run = AsyncMock(return_value=True)
    request = _write_only_request(feedback_repo)
    monkeypatch.setattr(feedback, "get_current_user", AsyncMock(return_value="user-1"))

    result = await feedback.delete_run_feedback(
        thread_id="thread-1",
        run_id="run-1",
        request=request,
    )

    assert result == {"success": True}
    feedback_repo.delete_by_run.assert_awaited_once_with(
        thread_id="thread-1",
        run_id="run-1",
        user_id="user-1",
    )


@pytest.mark.asyncio
async def test_write_permission_can_delete_feedback_record() -> None:
    feedback_repo = MagicMock()
    feedback_repo.get = AsyncMock(
        return_value={
            "feedback_id": "feedback-1",
            "thread_id": "thread-1",
            "run_id": "run-1",
        },
    )
    feedback_repo.delete = AsyncMock(return_value=True)
    request = _write_only_request(feedback_repo)

    result = await feedback.delete_feedback(
        thread_id="thread-1",
        run_id="run-1",
        feedback_id="feedback-1",
        request=request,
    )

    assert result == {"success": True}
    feedback_repo.delete.assert_awaited_once_with("feedback-1")


@pytest.mark.asyncio
async def test_write_permission_still_cannot_delete_thread_data() -> None:
    request = _write_only_request(MagicMock())

    with pytest.raises(HTTPException) as exc_info:
        await threads.delete_thread_data(thread_id="thread-1", request=request)

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "Permission denied: threads:delete"
