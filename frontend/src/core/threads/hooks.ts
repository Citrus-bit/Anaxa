import type { AIMessage, Message } from "@langchain/langgraph-sdk";
import type { ThreadsClient } from "@langchain/langgraph-sdk/client";
import { useStream } from "@langchain/langgraph-sdk/react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";

import type { PromptInputMessage } from "@/components/ai-elements/prompt-input";

import { getAPIClient } from "../api";
import { completeThreadRun, createRunEvent, registerThreadRun } from "../api/runs";
import { getBackendBaseURL } from "../config";
import { useI18n } from "../i18n/hooks";
import type { FileInMessage } from "../messages/utils";
import type { LocalSettings } from "../settings";
import { dispatchOpenSettings } from "../settings/events";
import { useUpdateSubtask } from "../tasks/context";
import type { UploadedFileInfo } from "../uploads";
import { uploadFiles } from "../uploads";

import type { AgentThread, AgentThreadState } from "./types";

export type ToolEndEvent = {
  name: string;
  data: unknown;
};

export type ThreadStreamOptions = {
  threadId?: string | null | undefined;
  context: LocalSettings["context"];
  isMock?: boolean;
  onStart?: (threadId: string, runId?: string) => void;
  onFinish?: (state: AgentThreadState) => void;
  onToolEnd?: (event: ToolEndEvent) => void;
};

export type ModelRetryStatus = {
  attempt: number;
  delaySeconds: number;
  errorClass?: string;
  statusCode?: number | null;
  providerCode?: string | null;
  retryAt?: string;
};

function getStreamErrorMessage(error: unknown): string {
  if (typeof error === "string" && error.trim()) {
    return error;
  }
  if (error instanceof Error && error.message.trim()) {
    return error.message;
  }
  if (typeof error === "object" && error !== null) {
    const message = Reflect.get(error, "message");
    if (typeof message === "string" && message.trim()) {
      return message;
    }
    const nestedError = Reflect.get(error, "error");
    if (nestedError instanceof Error && nestedError.message.trim()) {
      return nestedError.message;
    }
    if (typeof nestedError === "string" && nestedError.trim()) {
      return nestedError;
    }
  }
  return "Request failed.";
}

function getStreamErrorSignature(error: unknown): string {
  if (typeof error === "string") {
    return error.trim().toLowerCase();
  }
  if (error instanceof Error) {
    return `${error.name}:${error.message}`.toLowerCase();
  }
  if (typeof error === "object" && error !== null) {
    const parts = [
      Reflect.get(error, "name"),
      Reflect.get(error, "message"),
      Reflect.get(error, "status"),
      Reflect.get(error, "statusCode"),
      Reflect.get(error, "status_code"),
      Reflect.get(error, "code"),
      Reflect.get(error, "error"),
      Reflect.get(error, "body"),
    ]
      .map((value) => {
        if (typeof value === "string") return value;
        try {
          return JSON.stringify(value ?? "");
        } catch {
          return String(value ?? "");
        }
      })
      .join(" ");
    if (parts.trim()) {
      return parts.toLowerCase();
    }
  }
  return getStreamErrorMessage(error).toLowerCase();
}

const MODEL_PROVIDER_OVERLOAD_PATTERN =
  /(system_cpu_overloaded|system cpu overloaded|service unavailable|temporarily unavailable|temporarily overloaded|concurrency limit exceeded|please retry later|model unavailable|model is unavailable|provider unavailable|rate limit|too many requests|\b503\b|\b529\b|\b504\b|internalservererror|internal server error|an internal error occurred|overloaded)/i;

function isModelProviderOverloadedError(error: unknown): boolean {
  const signature = getStreamErrorSignature(error);
  return MODEL_PROVIDER_OVERLOAD_PATTERN.test(signature);
}

function isModelNotConfiguredError(error: unknown): boolean {
  const message = getStreamErrorMessage(error).toLowerCase();
  return (
    message.includes("no chat models are configured") ||
    message.includes("please configure at least one model")
  );
}

function getFriendlyStreamErrorMessage(
  error: unknown,
  overloadedMessage: string,
): string {
  if (isModelProviderOverloadedError(error)) {
    return overloadedMessage;
  }
  return getStreamErrorMessage(error);
}

const VISUAL_OUTPUT_INTENT_PATTERN =
  /(\b(chart|charts|graph|graphs|plot|plots|dashboard|ppt|pptx|slides?|presentation|image|images|illustration|diagram|flowchart|wireframe|ui|ux|frontend|website|webpage|landing page|visuali[sz]e|visualization|infographic)\b|图表|作图|画图|绘图|可视化|仪表盘|看板|幻灯片|演示文稿|PPT|图片|图像|插图|流程图|架构图|界面|前端|网页|网站|海报)/i;

function hasVisualOutputIntent(text: string): boolean {
  return VISUAL_OUTPUT_INTENT_PATTERN.test(text);
}

function compactValueSignature(value: unknown): string {
  const text =
    typeof value === "string"
      ? value
      : JSON.stringify(value ?? "", (_key, item) =>
          typeof item === "bigint" ? item.toString() : item,
        );
  return text.length > 500 ? `${text.slice(0, 500)}:${text.length}` : text;
}

export function threadStateSignature(values: AgentThreadState | null | undefined): string {
  if (!values) return "empty";
  const messages = Array.isArray(values.messages) ? values.messages : [];
  const artifacts = Array.isArray(values.artifacts) ? values.artifacts : [];
  const lastMessage = messages.at(-1);
  return JSON.stringify({
    title: values.title ?? "",
    updated_at: values.updated_at ?? "",
    status: values.status ?? "",
    message_count: messages.length,
    last_message_id: lastMessage?.id ?? "",
    last_message_type: lastMessage?.type ?? "",
    last_message_content: compactValueSignature(lastMessage?.content),
    artifact_count: artifacts.length,
    last_artifact: artifacts.at(-1) ?? "",
  });
}

export const STREAM_RENDER_COALESCE_MS = 80;

export type CoalesceDecision =
  | { action: "flush-now" }
  | { action: "schedule"; delayMs: number }
  | { action: "wait" };

export function decideCoalesce(
  nowMs: number,
  lastFlushMs: number,
  intervalMs: number,
  hasPendingTimer: boolean,
): CoalesceDecision {
  if (nowMs - lastFlushMs >= intervalMs) {
    return { action: "flush-now" };
  }
  if (hasPendingTimer) {
    return { action: "wait" };
  }
  return { action: "schedule", delayMs: intervalMs - (nowMs - lastFlushMs) };
}

function sameMessageArray(a: Message[], b: Message[]): boolean {
  return (
    a === b ||
    (a.length === b.length && a.every((message, index) => message === b[index]))
  );
}

/** Limit render-facing stream snapshots while keeping lifecycle state live. */
export function useCoalescedStreamMessages(
  messages: Message[],
  isStreaming: boolean,
  intervalMs: number = STREAM_RENDER_COALESCE_MS,
): Message[] {
  const [snapshot, setSnapshot] = useState<Message[] | null>(null);
  const latestRef = useRef(messages);
  latestRef.current = messages;
  const lastFlushRef = useRef(Number.NEGATIVE_INFINITY);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const publish = useCallback(() => {
    setSnapshot((previous) =>
      previous !== null && sameMessageArray(previous, latestRef.current)
        ? previous
        : latestRef.current,
    );
  }, []);

  const clearPendingFlush = useCallback(() => {
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (!isStreaming) {
      clearPendingFlush();
      lastFlushRef.current = Number.NEGATIVE_INFINITY;
      setSnapshot((previous) => (previous === null ? previous : null));
      return;
    }

    const now = performance.now();
    const decision = decideCoalesce(
      now,
      lastFlushRef.current,
      intervalMs,
      timerRef.current !== null,
    );
    if (decision.action === "flush-now") {
      clearPendingFlush();
      lastFlushRef.current = now;
      publish();
    } else if (decision.action === "schedule") {
      timerRef.current = setTimeout(() => {
        timerRef.current = null;
        lastFlushRef.current = performance.now();
        publish();
      }, decision.delayMs);
    }
  }, [messages, isStreaming, intervalMs, publish, clearPendingFlush]);

  useEffect(
    () => () => {
      if (timerRef.current !== null) {
        clearTimeout(timerRef.current);
      }
    },
    [],
  );

  return isStreaming && snapshot !== null ? snapshot : messages;
}

function getToolEventInput(data: unknown): Record<string, unknown> {
  if (typeof data === "object" && data !== null) {
    const input = Reflect.get(data, "input");
    if (typeof input === "object" && input !== null && !Array.isArray(input)) {
      return input as Record<string, unknown>;
    }
    if (typeof input === "string") {
      return { input };
    }
    const args = Reflect.get(data, "args");
    if (typeof args === "object" && args !== null && !Array.isArray(args)) {
      return args as Record<string, unknown>;
    }
  }
  return {};
}

function getToolEventRunId(event: unknown): string | undefined {
  if (typeof event !== "object" || event === null) return undefined;
  const runId = Reflect.get(event, "run_id");
  return typeof runId === "string" && runId.length > 0 ? runId : undefined;
}

function getModelRetryStatus(event: unknown): ModelRetryStatus | null {
  if (typeof event !== "object" || event === null) return null;
  if (Reflect.get(event, "type") !== "model_retry") return null;
  const attempt = Reflect.get(event, "attempt");
  const delaySeconds = Reflect.get(event, "delay_seconds");
  if (typeof attempt !== "number" || typeof delaySeconds !== "number") {
    return null;
  }
  const errorClass = Reflect.get(event, "error_class");
  const statusCode = Reflect.get(event, "status_code");
  const providerCode = Reflect.get(event, "provider_code");
  const retryAt = Reflect.get(event, "retry_at");
  return {
    attempt,
    delaySeconds,
    ...(typeof errorClass === "string" ? { errorClass } : {}),
    ...(typeof statusCode === "number" || statusCode === null
      ? { statusCode }
      : {}),
    ...(typeof providerCode === "string" || providerCode === null
      ? { providerCode }
      : {}),
    ...(typeof retryAt === "string" ? { retryAt } : {}),
  };
}

export function useThreadStream({
  threadId,
  context,
  isMock,
  onStart,
  onFinish,
  onToolEnd,
}: ThreadStreamOptions) {
  const { t } = useI18n();
  // Track the thread ID that is currently streaming to handle thread changes during streaming
  const [onStreamThreadId, setOnStreamThreadId] = useState(() => threadId);
  const [currentRunId, setCurrentRunId] = useState<string | null>(null);
  const [modelRetryStatus, setModelRetryStatus] =
    useState<ModelRetryStatus | null>(null);
  // Ref to track current thread ID across async callbacks without causing re-renders,
  // and to allow access to the current thread id in onUpdateEvent
  const threadIdRef = useRef<string | null>(threadId ?? null);
  const currentRunIdRef = useRef<string | null>(null);
  const startedRef = useRef(false);
  const recordedUserEventRef = useRef(false);
  const pendingUserContentRef = useRef("");
  const completedRunRef = useRef<{ runId: string; status: "success" | "error" } | null>(
    null,
  );

  const listeners = useRef({
    onStart,
    onFinish,
    onToolEnd,
  });

  // Keep listeners ref updated with latest callbacks
  useEffect(() => {
    listeners.current = { onStart, onFinish, onToolEnd };
  }, [onStart, onFinish, onToolEnd]);

  useEffect(() => {
    const normalizedThreadId = threadId ?? null;
    if (!normalizedThreadId) {
      // Just reset for new thread creation when threadId becomes null/undefined
      startedRef.current = false;
      setOnStreamThreadId(normalizedThreadId);
      setCurrentRunId(null);
      setModelRetryStatus(null);
      currentRunIdRef.current = null;
      completedRunRef.current = null;
    } else if (normalizedThreadId !== threadIdRef.current) {
      setCurrentRunId(null);
      setModelRetryStatus(null);
      currentRunIdRef.current = null;
      recordedUserEventRef.current = false;
      completedRunRef.current = null;
    }
    threadIdRef.current = normalizedThreadId;
  }, [threadId]);

  const _handleOnStart = useCallback((id: string, runId?: string) => {
    if (!startedRef.current) {
      listeners.current.onStart?.(id, runId);
      startedRef.current = true;
    }
  }, []);

  const handleStreamStart = useCallback(
    (_threadId: string, runId?: string) => {
      threadIdRef.current = _threadId;
      _handleOnStart(_threadId, runId);
    },
    [_handleOnStart],
  );

  const queryClient = useQueryClient();
  const updateSubtask = useUpdateSubtask();

  const completeCurrentRun = useCallback((status: "success" | "error") => {
    const runId = currentRunIdRef.current;
    const currentThreadId = threadIdRef.current;
    if (!runId || !currentThreadId) {
      return;
    }
    if (
      completedRunRef.current?.runId === runId &&
      completedRunRef.current.status === status
    ) {
      return;
    }
    completedRunRef.current = { runId, status };
    void completeThreadRun(currentThreadId, runId, status).catch(() => undefined);
  }, []);

  const showStreamErrorToast = useCallback(
    (error: unknown) => {
      if (isModelProviderOverloadedError(error)) {
        toast.error(t.conversation.modelProviderOverloaded, {
          id: "model-provider-overloaded",
        });
        return;
      }
      toast.error(getFriendlyStreamErrorMessage(error, t.conversation.modelProviderOverloaded));
    },
    [t.conversation.modelProviderOverloaded],
  );

  const thread = useStream<AgentThreadState>({
    client: getAPIClient(isMock),
    assistantId: "lead_agent",
    threadId: onStreamThreadId,
    reconnectOnMount: true,
    fetchStateHistory: { limit: 1 },
    throttle: true,
    onCreated(meta) {
      handleStreamStart(meta.thread_id, meta.run_id);
      setCurrentRunId(meta.run_id);
      setModelRetryStatus(null);
      currentRunIdRef.current = meta.run_id;
      completedRunRef.current = null;
      setOnStreamThreadId(meta.thread_id);
      void registerThreadRun(meta.thread_id, meta.run_id, {
        assistantId:
          typeof context.agent_name === "string" && context.agent_name.length > 0
            ? context.agent_name
            : "lead_agent",
        context,
      }).catch(() => {
        // Best-effort only. Older backends should not break streaming.
      });
      if (!recordedUserEventRef.current) {
        recordedUserEventRef.current = true;
        const content = pendingUserContentRef.current;
        void createRunEvent(meta.thread_id, meta.run_id, {
          event_type: "human_message",
          caller: "user",
          content: { type: "human", content },
        }).catch(() => undefined);
      }
      // Optimistically add the new thread to the sidebar list so it's visible
      // immediately, even before the backend's threads.search returns it.
      queryClient.setQueriesData(
        { queryKey: ["threads", "search"], exact: false },
        (oldData: Array<AgentThread> | undefined) => {
          if (!oldData) return oldData;
          // Avoid duplicates
          if (oldData.some((t) => t.thread_id === meta.thread_id)) {
            return oldData;
          }
          const optimisticThread: AgentThread = {
            thread_id: meta.thread_id,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            state_updated_at: new Date().toISOString(),
            metadata: {},
            status: "busy",
            values: { title: "", messages: [], artifacts: [] },
            interrupts: {},
          };
          return [optimisticThread, ...oldData];
        },
      );
    },
    onLangChainEvent(event) {
      if (event.event === "on_chat_model_stream" || event.event === "on_chat_model_end") {
        setModelRetryStatus(null);
      }
      if (event.event === "on_tool_start") {
        const runId = currentRunIdRef.current;
        const currentThreadId = threadIdRef.current;
        if (runId && currentThreadId) {
          void createRunEvent(currentThreadId, runId, {
            event_type: "ai_tool_calls",
            caller: "assistant",
            content: {
              type: "ai",
              tool_calls: [
                {
                  name: event.name || "tool",
                  args: getToolEventInput(event.data),
                  id: getToolEventRunId(event) ?? `${event.name || "tool"}-start`,
                },
              ],
            },
          }).catch(() => undefined);
        }
      }
      if (event.event === "on_tool_end") {
        listeners.current.onToolEnd?.({
          name: event.name,
          data: event.data,
        });
        const runId = currentRunIdRef.current;
        const currentThreadId = threadIdRef.current;
        if (runId && currentThreadId) {
          void createRunEvent(currentThreadId, runId, {
            event_type: "tool_message",
            caller: event.name || "tool",
            content: {
              type: "tool",
              name: event.name,
              content:
                typeof event.data === "string"
                  ? event.data
                  : JSON.stringify(event.data ?? {}),
            },
          }).catch(() => undefined);
        }
      }
    },
    onUpdateEvent(data) {
      const updates: Array<Partial<AgentThreadState> | null> = Object.values(
        data || {},
      );
      for (const update of updates) {
        if (update && "title" in update && update.title) {
          void queryClient.setQueriesData(
            {
              queryKey: ["threads", "search"],
              exact: false,
            },
            (oldData: Array<AgentThread> | undefined) => {
              return oldData?.map((t) => {
                if (t.thread_id === threadIdRef.current) {
                  return {
                    ...t,
                    values: {
                      ...t.values,
                      title: update.title,
                    },
                  };
                }
                return t;
              });
            },
          );
        }
        if (update && "artifacts" in update && Array.isArray(update.artifacts)) {
          const runId = currentRunIdRef.current;
          const currentThreadId = threadIdRef.current;
          if (runId && currentThreadId) {
            void createRunEvent(currentThreadId, runId, {
              event_type: "state_snapshot",
              caller: "checkpoint",
              content: { artifacts: update.artifacts },
            }).catch(() => undefined);
          }
        }
      }
    },
    onCustomEvent(event: unknown) {
      const modelRetry = getModelRetryStatus(event);
      if (modelRetry) {
        setModelRetryStatus(modelRetry);
        const runId = currentRunIdRef.current;
        const currentThreadId = threadIdRef.current;
        if (runId && currentThreadId) {
          void createRunEvent(currentThreadId, runId, {
            event_type: "model_retry",
            caller: "model",
            content: {
              type: "model_retry",
              attempt: modelRetry.attempt,
              delay_seconds: modelRetry.delaySeconds,
              error_class: modelRetry.errorClass,
              status_code: modelRetry.statusCode,
              provider_code: modelRetry.providerCode,
              retry_at: modelRetry.retryAt,
            },
          }).catch(() => undefined);
        }
        return;
      }
      if (
        typeof event === "object" &&
        event !== null &&
        "type" in event &&
        event.type === "task_running"
      ) {
        const e = event as {
          type: "task_running";
          task_id?: string;
          message?: AIMessage;
          heartbeat?: boolean;
        };
        if (!e.task_id) {
          return;
        }
        updateSubtask({
          id: e.task_id,
          ...(e.message ? { latestMessage: e.message } : {}),
          lastUpdatedAt: new Date().toISOString(),
        });
        if (e.heartbeat && !e.message) {
          return;
        }
        const runId = currentRunIdRef.current;
        const currentThreadId = threadIdRef.current;
        if (runId && currentThreadId) {
          void createRunEvent(currentThreadId, runId, {
            event_type: "subagent_event",
            caller: "task",
            content: {
              type: "task",
              task_id: e.task_id,
              heartbeat: Boolean(e.heartbeat),
              content:
                typeof e.message?.content === "string"
                  ? e.message.content
                  : JSON.stringify(e.message?.content ?? ""),
            },
          }).catch(() => undefined);
        }
      }
    },
    onError(error) {
      setOptimisticMessages([]);
      setModelRetryStatus(null);
      completeCurrentRun("error");
      if (isModelNotConfiguredError(error)) {
        toast.error(t.setup.noModelsConfigured, {
          id: "model-not-configured",
          description: t.setup.noModelsConfiguredHint,
          duration: 8000,
          action: {
            label: t.setup.openSettings,
            onClick: () => dispatchOpenSettings({ section: "setup" }),
          },
        });
        return;
      }
      showStreamErrorToast(error);
    },
    onFinish(state) {
      setModelRetryStatus(null);
      completeCurrentRun("success");
      listeners.current.onFinish?.(state.values);
      void queryClient.invalidateQueries({ queryKey: ["threads", "search"] });
    },
  });

  const [optimisticMessages, setOptimisticMessages] = useState<Message[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const sendInFlightRef = useRef(false);
  const prevMsgCountRef = useRef(thread.messages.length);
  const polledSignatureRef = useRef<string | null>(null);
  const [polledValues, setPolledValues] = useState<AgentThreadState | null>(
    null,
  );

  const liveMessages =
    polledValues !== null && Array.isArray(polledValues.messages)
      ? polledValues.messages
      : thread.messages;
  const renderedMessages = useCoalescedStreamMessages(
    liveMessages,
    thread.isLoading,
  );

  useEffect(() => {
    if (thread.isLoading) {
      setIsSubmitting(false);
    }
  }, [thread.isLoading]);

  useEffect(() => {
    if (!thread.isLoading) {
      setPolledValues(null);
      polledSignatureRef.current = null;
      return;
    }

    const client = getAPIClient(isMock);
    let cancelled = false;

    const syncThreadState = async () => {
      if (document.visibilityState !== "visible") return;
      const currentThreadId = threadIdRef.current;
      if (!currentThreadId) return;
      try {
        const state = await client.threads.getState<AgentThreadState>(
          currentThreadId,
        );
        if (cancelled) return;
        const nextSignature = threadStateSignature(state.values);
        if (nextSignature === polledSignatureRef.current) {
          return;
        }
        polledSignatureRef.current = nextSignature;
        setPolledValues(state.values);
        void queryClient.invalidateQueries({ queryKey: ["threads", "search"] });
      } catch {
        // Best-effort fallback only.
      }
    };

    void syncThreadState();
    const interval = window.setInterval(() => {
      void syncThreadState();
    }, 5000);

    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [isMock, queryClient, thread.isLoading]);

  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.visibilityState === "visible") {
        void queryClient.invalidateQueries({
          queryKey: ["threads", "search"],
        });
      }
    };
    document.addEventListener("visibilitychange", handleVisibilityChange);
    return () => {
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [queryClient]);

  // Clear optimistic when server messages arrive (count increases)
  useEffect(() => {
    if (
      optimisticMessages.length > 0 &&
      thread.messages.length > prevMsgCountRef.current
    ) {
      setOptimisticMessages([]);
    }
  }, [thread.messages.length, optimisticMessages.length]);

  const snapshotThread = {
    ...thread,
    ...(polledValues === null ? {} : { values: polledValues }),
    messages: renderedMessages,
  } as typeof thread;

  const sendMessage = useCallback(
    async (
      threadId: string,
      message: PromptInputMessage,
      extraContext?: Record<string, unknown>,
    ) => {
      if (sendInFlightRef.current) {
        try {
          await thread.stop();
        } catch {
          // stop may fail if the run already finished — safe to ignore
        }
        sendInFlightRef.current = false;
      }
      sendInFlightRef.current = true;
      setIsSubmitting(true);
      recordedUserEventRef.current = false;

      const text = message.text.trim();
      pendingUserContentRef.current = text;

      // Capture current count before showing optimistic messages
      prevMsgCountRef.current = thread.messages.length;

      // Build optimistic files list with uploading status
      const optimisticFiles: FileInMessage[] = (message.files ?? []).map(
        (f) => ({
          filename: f.filename ?? "",
          size: 0,
          status: "uploading" as const,
        }),
      );

      // Create optimistic human message (shown immediately)
      const optimisticHumanMsg: Message = {
        type: "human",
        id: `opt-human-${Date.now()}`,
        content: text ? [{ type: "text", text }] : "",
        additional_kwargs:
          optimisticFiles.length > 0 ? { files: optimisticFiles } : {},
      };

      const newOptimistic: Message[] = [optimisticHumanMsg];

      // Show an optimistic "thinking" AI placeholder so the user sees immediate
      // feedback while the request reaches the backend and streaming begins.
      if (optimisticFiles.length > 0) {
        newOptimistic.push({
          type: "ai",
          id: `opt-ai-upload-${Date.now()}`,
          content: t.uploads.uploadingFiles,
          additional_kwargs: { element: "task" },
        });
      } else if (text) {
        newOptimistic.push({
          type: "ai",
          id: `opt-ai-thinking-${Date.now()}`,
          content: t.common.thinking,
          additional_kwargs: { _thinking: true },
        });
      }

      setOptimisticMessages(newOptimistic);
      setIsSubmitting(true);

      _handleOnStart(threadId);

      let uploadedFileInfo: UploadedFileInfo[] = [];

      try {
        // Upload files first if any
        if (message.files && message.files.length > 0) {
          setIsUploading(true);
          try {
            // Convert FileUIPart to File objects by fetching blob URLs
            const filePromises = message.files.map(async (fileUIPart) => {
              if (fileUIPart.url && fileUIPart.filename) {
                try {
                  // Fetch the blob URL to get the file data
                  const response = await fetch(fileUIPart.url);
                  const blob = await response.blob();

                  // Create a File object from the blob
                  return new File([blob], fileUIPart.filename, {
                    type: fileUIPart.mediaType || blob.type,
                  });
                } catch (error) {
                  console.error(
                    `Failed to fetch file ${fileUIPart.filename}:`,
                    error,
                  );
                  return null;
                }
              }
              return null;
            });

            const conversionResults = await Promise.all(filePromises);
            const files = conversionResults.filter(
              (file): file is File => file !== null,
            );
            const failedConversions = conversionResults.length - files.length;

            if (failedConversions > 0) {
              throw new Error(
                `Failed to prepare ${failedConversions} attachment(s) for upload. Please retry.`,
              );
            }

            if (!threadId) {
              throw new Error("Thread is not ready for file upload.");
            }

            if (files.length > 0) {
              const uploadResponse = await uploadFiles(threadId, files);
              uploadedFileInfo = uploadResponse.files;

              // Update optimistic human message with uploaded status + paths
              const uploadedFiles: FileInMessage[] = uploadedFileInfo.map(
                (info) => ({
                  filename: info.filename,
                  size: info.size,
                  path: info.virtual_path,
                  status: "uploaded" as const,
                }),
              );
              setOptimisticMessages((messages) => {
                if (messages.length > 1 && messages[0]) {
                  const humanMessage: Message = messages[0];
                  return [
                    {
                      ...humanMessage,
                      additional_kwargs: { files: uploadedFiles },
                    },
                    ...messages.slice(1),
                  ];
                }
                return messages;
              });
            }
          } catch (error) {
            console.error("Failed to upload files:", error);
            const errorMessage =
              error instanceof Error
                ? error.message
                : "Failed to upload files.";
            toast.error(errorMessage);
            setOptimisticMessages([]);
            throw error;
          } finally {
            setIsUploading(false);
          }
        }

        // Build files metadata for submission (included in additional_kwargs)
        const filesForSubmit: FileInMessage[] = uploadedFileInfo.map(
          (info) => ({
            filename: info.filename,
            size: info.size,
            path: info.virtual_path,
            status: "uploaded" as const,
          }),
        );

        await thread.submit(
          {
            messages: [
              {
                type: "human",
                content: [
                  {
                    type: "text",
                    text,
                  },
                ],
                additional_kwargs:
                  filesForSubmit.length > 0 ? { files: filesForSubmit } : {},
              },
            ],
          },
          {
            threadId: threadId,
            streamSubgraphs: true,
            streamResumable: true,
            config: {
              recursion_limit: 1000,
            },
            context: {
              ...extraContext,
              ...context,
              thinking_enabled: context.mode !== "flash",
              is_plan_mode: context.mode === "pro" || context.mode === "ultra",
              subagent_enabled: context.mode === "ultra",
              visual_output_intent:
                Boolean(extraContext?.visual_output_intent) ||
                Boolean(context.visual_output_intent) ||
                hasVisualOutputIntent(text),
              synthetic_data_mode:
                Boolean(extraContext?.synthetic_data_mode) ||
                Boolean(context.synthetic_data_mode),
              reasoning_effort:
                context.reasoning_effort ??
                (context.mode === "ultra"
                  ? "xhigh"
                  : context.mode === "pro"
                    ? "high"
                    : "medium"),
              thread_id: threadId,
            },
          },
        );
        void queryClient.invalidateQueries({ queryKey: ["threads", "search"] });
      } catch (error) {
        setOptimisticMessages([]);
        setIsUploading(false);
        setIsSubmitting(false);
        if (isModelProviderOverloadedError(error)) {
          completeCurrentRun("error");
          showStreamErrorToast(error);
          void queryClient.invalidateQueries({ queryKey: ["threads", "search"] });
          return;
        }
        throw error;
      } finally {
        sendInFlightRef.current = false;
      }
    },
    [
      thread,
      _handleOnStart,
      t.uploads.uploadingFiles,
      t.common.thinking,
      context,
      queryClient,
      completeCurrentRun,
      showStreamErrorToast,
    ],
  );

  // Merge thread with optimistic messages for display
  const mergedThread =
    optimisticMessages.length > 0
      ? ({
          ...snapshotThread,
          messages: [...(snapshotThread.messages ?? []), ...optimisticMessages],
        } as typeof thread)
      : snapshotThread;

  return [
    mergedThread,
    sendMessage,
    isUploading,
    isSubmitting,
    currentRunId,
    modelRetryStatus,
  ] as const;
}

export function useThreads(
  params: Parameters<ThreadsClient["search"]>[0] = {
    limit: 50,
    sortBy: "updated_at",
    sortOrder: "desc",
    select: ["thread_id", "updated_at", "values", "status"],
  },
) {
  const apiClient = getAPIClient();
  return useQuery<AgentThread[]>({
    queryKey: ["threads", "search", params],
    queryFn: async () => {
      const maxResults = params.limit;
      const initialOffset = params.offset ?? 0;
      const DEFAULT_PAGE_SIZE = 50;

      // Preserve prior semantics: if a non-positive limit is explicitly provided,
      // delegate to a single search call with the original parameters.
      if (maxResults !== undefined && maxResults <= 0) {
        const response = await apiClient.threads.search<AgentThreadState>(params);
        return response as AgentThread[];
      }

      const pageSize =
        typeof maxResults === "number" && maxResults > 0
          ? Math.min(DEFAULT_PAGE_SIZE, maxResults)
          : DEFAULT_PAGE_SIZE;

      const threads: AgentThread[] = [];
      let offset = initialOffset;

      while (true) {
        if (typeof maxResults === "number" && threads.length >= maxResults) {
          break;
        }

        const currentLimit =
          typeof maxResults === "number"
            ? Math.min(pageSize, maxResults - threads.length)
            : pageSize;

        if (typeof maxResults === "number" && currentLimit <= 0) {
          break;
        }

        const response = (await apiClient.threads.search<AgentThreadState>({
          ...params,
          limit: currentLimit,
          offset,
        })) as AgentThread[];

        threads.push(...response);

        if (response.length < currentLimit) {
          break;
        }

        offset += response.length;
      }

      return threads;
    },
    refetchOnWindowFocus: true,
    staleTime: 30_000,
  });
}

export function useDeleteThread() {
  const queryClient = useQueryClient();
  const apiClient = getAPIClient();
  return useMutation({
    mutationFn: async ({ threadId }: { threadId: string }) => {
      await apiClient.threads.delete(threadId);

      const response = await fetch(
        `${getBackendBaseURL()}/api/threads/${encodeURIComponent(threadId)}`,
        {
          method: "DELETE",
        },
      );

      if (!response.ok) {
        const error = await response
          .json()
          .catch(() => ({ detail: "Failed to delete local thread data." }));
        throw new Error(error.detail ?? "Failed to delete local thread data.");
      }
    },
    onSuccess(_, { threadId }) {
      queryClient.setQueriesData(
        {
          queryKey: ["threads", "search"],
          exact: false,
        },
        (oldData: Array<AgentThread> | undefined) => {
          if (oldData == null) {
            return oldData;
          }
          return oldData.filter((t) => t.thread_id !== threadId);
        },
      );
    },
    onError(error) {
      toast.error(getStreamErrorMessage(error));
    },
    onSettled() {
      void queryClient.invalidateQueries({ queryKey: ["threads", "search"] });
    },
  });
}

export function useRenameThread() {
  const queryClient = useQueryClient();
  const apiClient = getAPIClient();
  return useMutation({
    mutationFn: async ({
      threadId,
      title,
    }: {
      threadId: string;
      title: string;
    }) => {
      await apiClient.threads.updateState(threadId, {
        values: { title },
      });
    },
    onSuccess(_, { threadId, title }) {
      queryClient.setQueriesData(
        {
          queryKey: ["threads", "search"],
          exact: false,
        },
        (oldData: Array<AgentThread>) => {
          return oldData.map((t) => {
            if (t.thread_id === threadId) {
              return {
                ...t,
                values: {
                  ...t.values,
                  title,
                },
              };
            }
            return t;
          });
        },
      );
    },
    onError(error) {
      toast.error(getStreamErrorMessage(error));
    },
  });
}
