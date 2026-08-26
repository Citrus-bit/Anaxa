import type { Message } from "@langchain/langgraph-sdk";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  extractContentFromMessage,
  extractReasoningContentFromMessage,
  findClarificationResponse,
  groupMessages,
  hasReasoning,
} from "./utils";

describe("groupMessages", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("silently ignores orphan tool messages from partial stream replay", () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined);
    const messages = [
      {
        id: "tool-1",
        type: "tool",
        name: "citation_audit",
        tool_call_id: "tc-1",
        content: "PASS",
      },
    ] as unknown as Message[];

    const groups = groupMessages(messages, (group) => group.type);

    expect(groups).toEqual([]);
    expect(errorSpy).not.toHaveBeenCalled();
  });

  it("attaches tool messages by tool_call_id even after a terminal assistant message", () => {
    const messages = [
      {
        id: "ai-1",
        type: "ai",
        content: "",
        tool_calls: [
          {
            id: "tc-1",
            name: "citation_audit",
            args: {},
          },
        ],
      },
      {
        id: "ai-2",
        type: "ai",
        content: "Working from the audit result.",
      },
      {
        id: "tool-1",
        type: "tool",
        name: "citation_audit",
        tool_call_id: "tc-1",
        content: "PASS",
      },
    ] as unknown as Message[];

    const groups = groupMessages(messages, (group) => ({
      type: group.type,
      count: group.messages.length,
    }));

    expect(groups).toEqual([
      { type: "assistant:processing", count: 2 },
      { type: "assistant", count: 1 },
    ]);
  });
});

describe("findClarificationResponse", () => {
  it("returns the first human response after a clarification message", () => {
    const clarificationMessage = {
      id: "tool-1",
      type: "tool",
      name: "ask_clarification",
      tool_call_id: "tool-1",
      content: "Which template?",
      additional_kwargs: {
        clarification: {
          question: "Which template?",
          options: ["Option A", "Option B"],
        },
      },
    } as const;

    const response = findClarificationResponse(
      [
        { id: "ai-1", type: "ai", content: "Need clarification." },
        clarificationMessage,
        { id: "human-1", type: "human", content: "Option A" },
        { id: "ai-2", type: "ai", content: "Continuing..." },
      ],
      clarificationMessage,
    );

    expect(response).toBe("Option A");
  });

  it("stops scanning at the next clarification message", () => {
    const firstClarification = {
      id: "tool-1",
      type: "tool",
      name: "ask_clarification",
      tool_call_id: "tool-1",
      content: "First question",
    } as const;
    const secondClarification = {
      id: "tool-2",
      type: "tool",
      name: "ask_clarification",
      tool_call_id: "tool-2",
      content: "Second question",
    } as const;

    const response = findClarificationResponse(
      [
        firstClarification,
        { id: "ai-1", type: "ai", content: "Still pending" },
        secondClarification,
        { id: "human-1", type: "human", content: "Reply to second" },
      ],
      firstClarification,
    );

    expect(response).toBeNull();
  });
});

describe("inline reasoning extraction", () => {
  it("reuses a split for unchanged message content", () => {
    const message = {
      type: "ai",
      content: "<think>plan</think>Answer",
    } as unknown as Message;

    expect(extractContentFromMessage(message)).toBe("Answer");
    expect(extractReasoningContentFromMessage(message)).toBe("plan");
    expect(hasReasoning(message)).toBe(true);
  });

  it("invalidates the cache when a streaming message is mutated", () => {
    const message = {
      type: "ai",
      content: "<think>first</think>Answer",
    } as unknown as Message;

    expect(extractReasoningContentFromMessage(message)).toBe("first");
    message.content = "<think>second</think>Updated";

    expect(extractContentFromMessage(message)).toBe("Updated");
    expect(extractReasoningContentFromMessage(message)).toBe("second");
  });
});
