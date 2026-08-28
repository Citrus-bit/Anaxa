import type { Message } from "@langchain/langgraph-sdk";

import type { AgentThread, AgentThreadContext, AgentThreadState } from "./types";

// Namespaced to match other internal metadata keys (``deerflow_sidecar``,
// ``deerflow_branch``) so it cannot collide with a future feature or a
// client-supplied key. Keep in sync with the backend thread_meta constant and
// the E2E mock-api constant.
export const THREAD_PINNED_METADATA_KEY = "deerflow_pinned";

export type ChannelThreadSource = {
  type: "im_channel";
  provider: string;
  label: string;
};

type ThreadRouteTarget =
  | string
  | {
      thread_id: string;
      context?: Pick<AgentThreadContext, "agent_name"> | null;
      metadata?: Record<string, unknown> | null;
    };

export function pathOfThread(
  thread: ThreadRouteTarget,
  context?: Pick<AgentThreadContext, "agent_name"> | null,
) {
  const threadId = typeof thread === "string" ? thread : thread.thread_id;
  const encodedThreadId = encodeURIComponent(threadId);
  let agentName: string | undefined;
  if (typeof thread === "string") {
    agentName = context?.agent_name;
  } else {
    agentName = thread.context?.agent_name;
    if (!agentName) {
      const metaAgent = thread.metadata?.agent_name;
      if (typeof metaAgent === "string") {
        agentName = metaAgent;
      }
    }
  }

  return agentName
    ? `/workspace/agents/${encodeURIComponent(agentName)}/chats/${encodedThreadId}`
    : `/workspace/chats/${encodedThreadId}`;
}

export function textOfMessage(message: Message) {
  if (typeof message.content === "string") {
    return message.content;
  } else if (Array.isArray(message.content)) {
    // Flat join ("") for single-line consumers (input box, titles); the rendered
    // body uses extractContentFromMessage, which joins multi-part content with "\n".
    const text = message.content
      .map((part) =>
        typeof part === "string" ? part : part.type === "text" ? part.text : "",
      )
      .join("");
    return text.length > 0 ? text : null;
  }
  return null;
}

const THINK_BLOCK_RE = /<think\b[^>]*>[\s\S]*?<\/think>/gi;
const THINK_OPEN_RE = /<think\b[^>]*>/i;
const TITLE_LABEL_RE = /^(?:title|标题)\s*[:：-]\s*/i;
const TITLE_INTRO_RE =
  /^(?:here(?:'s| is)\s+(?:the\s+)?title|the\s+title\s+is)\s*[:：-]\s*/i;
const TITLE_PROMPT_ECHO_RE =
  /generate a concise title|return only the title|user message:|assistant summary:|^the user\b|^the assistant\b/i;
const GENERIC_SUMMARY_TITLE_RE =
  /^(?:here(?:'s| is)\s+(?:a\s+)?(?:brief\s+|concise\s+)?summary\s+of\s+(?:the\s+)?(?:conversation|chat)(?:\s+to\s+date)?|(?:the\s+)?(?:conversation|chat)\s+summary|summary\s+of\s+(?:this|the)\s+(?:conversation|chat)|(?:本次|这次|当前)?(?:对话|聊天)(?:的)?(?:总结|小结))\s*[:：。.!！-]*$/i;
const TITLE_MAX_CHARS = 90;

function normalizeInlineText(text: string) {
  return text
    .replace(/\s+/g, " ")
    .trim()
    .replace(/^[`"'“”‘’]+|[`"'“”‘’]+$/g, "")
    .trim();
}

function truncateTitle(title: string) {
  const clean = normalizeInlineText(title);
  if (clean.length <= TITLE_MAX_CHARS) return clean;
  return `${clean.slice(0, TITLE_MAX_CHARS).trimEnd()}...`;
}

/** Clean model-generated title noise without changing user-authored text. */
export function cleanThreadTitle(title: string | null | undefined) {
  if (!title) return "";

  const withoutClosedThinking = title.replace(THINK_BLOCK_RE, "\n");
  if (THINK_OPEN_RE.test(withoutClosedThinking)) {
    return "";
  }

  const lines = withoutClosedThinking
    .split(/\r?\n/)
    .map((line) =>
      normalizeInlineText(
        line
          .replace(TITLE_INTRO_RE, "")
          .replace(TITLE_LABEL_RE, "")
          .replace(/^[*#>\-\s]+/, ""),
      ),
    )
    .filter(Boolean);

  for (const line of lines) {
    if (
      TITLE_PROMPT_ECHO_RE.test(line) ||
      GENERIC_SUMMARY_TITLE_RE.test(line) ||
      THINK_OPEN_RE.test(line)
    ) {
      continue;
    }
    return truncateTitle(line);
  }

  return "";
}

function fallbackTitleFromMessages(messages: Message[] | undefined) {
  if (!messages) return "Untitled";
  for (const message of messages) {
    if (message.type !== "human") continue;
    const text = textOfMessage(message);
    if (!text) continue;
    const firstLine = text.split(/\r?\n/).find((line) => line.trim());
    if (firstLine) {
      return truncateTitle(firstLine);
    }
  }
  return "Untitled";
}

export function titleOfThreadState(
  state: Pick<AgentThreadState, "title" | "messages"> | null | undefined,
) {
  if (!state) return "Untitled";
  return cleanThreadTitle(state.title) || fallbackTitleFromMessages(state.messages);
}

export function titleOfThread(thread: AgentThread) {
  return titleOfThreadState(thread.values);
}

export function isThreadPinned(thread: Pick<AgentThread, "metadata">) {
  return thread.metadata?.[THREAD_PINNED_METADATA_KEY] === true;
}

export function sortPinnedThreads<T extends Pick<AgentThread, "metadata">>(
  threads: readonly T[],
) {
  return threads
    .map((thread, index) => ({ thread, index }))
    .sort((left, right) => {
      const pinnedDiff =
        Number(isThreadPinned(right.thread)) -
        Number(isThreadPinned(left.thread));
      return pinnedDiff || left.index - right.index;
    })
    .map(({ thread }) => thread);
}

const CHANNEL_PROVIDER_LABELS: Record<string, string> = {
  buzz: "Buzz",
  dingtalk: "DingTalk",
  discord: "Discord",
  feishu: "Feishu",
  slack: "Slack",
  telegram: "Telegram",
  wechat: "WeChat",
  wecom: "WeCom",
};

function labelOfChannelProvider(provider: string) {
  return CHANNEL_PROVIDER_LABELS[provider] ?? provider;
}

export function channelSourceOfThread(
  thread: Pick<AgentThread, "metadata">,
): ChannelThreadSource | null {
  const source = thread.metadata?.channel_source;
  if (!source || typeof source !== "object" || Array.isArray(source)) {
    return null;
  }

  if (Reflect.get(source, "type") !== "im_channel") {
    return null;
  }

  const provider = Reflect.get(source, "provider");
  if (typeof provider !== "string" || provider.trim().length === 0) {
    return null;
  }

  const normalizedProvider = provider.trim().toLowerCase();
  return {
    type: "im_channel",
    provider: normalizedProvider,
    label: labelOfChannelProvider(normalizedProvider),
  };
}
