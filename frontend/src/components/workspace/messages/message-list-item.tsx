import type { Message } from "@langchain/langgraph-sdk";
import { FileIcon, Loader2Icon, ThumbsDownIcon, ThumbsUpIcon } from "lucide-react";
import { useParams } from "next/navigation";
import { memo, useEffect, useMemo, useState, type ImgHTMLAttributes } from "react";

import { Loader } from "@/components/ai-elements/loader";
import {
  Message as AIElementMessage,
  MessageContent as AIElementMessageContent,
  MessageResponse as AIElementMessageResponse,
  MessageToolbar,
} from "@/components/ai-elements/message";
import {
  Reasoning,
  ReasoningContent,
  ReasoningTrigger,
} from "@/components/ai-elements/reasoning";
import { Task, TaskTrigger } from "@/components/ai-elements/task";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  deleteRunFeedback,
  getRunFeedback,
  putRunFeedback,
  type FeedbackData,
} from "@/core/api/feedback";
import { resolveArtifactURL } from "@/core/artifacts/utils";
import { useI18n } from "@/core/i18n/hooks";
import {
  extractContentFromMessage,
  extractReasoningContentFromMessage,
  parseUploadedFiles,
  stripUploadedFilesTag,
  type FileInMessage,
} from "@/core/messages/utils";
import { useRehypeSplitWordsIntoSpans } from "@/core/rehype";
import { humanMessagePlugins } from "@/core/streamdown";
import { cn } from "@/lib/utils";

import { CopyButton } from "../copy-button";

import { shouldShowRunFeedback } from "./feedback-visibility";
import { MarkdownContent } from "./markdown-content";

export function MessageListItem({
  className,
  message,
  isLoading,
  runId,
  threadId,
}: {
  className?: string;
  message: Message;
  isLoading?: boolean;
  runId?: string | null;
  threadId: string;
}) {
  const isHuman = message.type === "human";
  const showFeedback = shouldShowRunFeedback({
    messageType: message.type,
    isLoading,
    runId,
  });
  return (
    <AIElementMessage
      className={cn("group/conversation-message relative w-full", className)}
      data-testid={
        !isHuman && runId ? "assistant-message-with-feedback" : undefined
      }
      from={isHuman ? "user" : "assistant"}
    >
      <MessageContent
        className={isHuman ? "w-fit" : "w-full"}
        message={message}
        isLoading={isLoading}
      />
      {!isLoading && (
        <MessageToolbar
          className={cn(
            isHuman ? "-bottom-9 justify-end" : "-bottom-8",
            "absolute right-0 left-0 z-20 opacity-0 transition-opacity delay-200 duration-300 group-hover/conversation-message:opacity-100",
          )}
        >
          <div className="flex gap-1">
            <CopyButton
              clipboardData={
                extractContentFromMessage(message) ??
                extractReasoningContentFromMessage(message) ??
                ""
              }
            />
            {showFeedback && runId && (
              <MessageFeedbackControls threadId={threadId} runId={runId} />
            )}
          </div>
        </MessageToolbar>
      )}
    </AIElementMessage>
  );
}

function MessageFeedbackControls({
  threadId,
  runId,
}: {
  threadId: string;
  runId: string;
}) {
  const [feedback, setFeedback] = useState<FeedbackData | null>(null);
  const [supported, setSupported] = useState(true);
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setSupported(true);
    setFeedback(null);

    void getRunFeedback(threadId, runId)
      .then((value) => {
        if (!cancelled) {
          setFeedback(value);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setSupported(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [threadId, runId]);

  if (!supported) {
    return null;
  }

  const handleRate = async (rating: 1 | -1) => {
    if (isSaving) {
      return;
    }
    setIsSaving(true);
    try {
      if (feedback?.rating === rating) {
        await deleteRunFeedback(threadId, runId);
        setFeedback(null);
      } else {
        setFeedback(await putRunFeedback(threadId, runId, rating));
      }
    } catch {
      setSupported(false);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <>
      <Button
        size="icon-sm"
        type="button"
        variant="ghost"
        data-testid="message-feedback-up"
        aria-label="Thumbs up"
        aria-pressed={feedback?.rating === 1}
        disabled={isSaving}
        className={cn(feedback?.rating === 1 && "text-primary bg-primary/10 hover:bg-primary/15")}
        onClick={() => void handleRate(1)}
      >
        <ThumbsUpIcon size={12} />
      </Button>
      <Button
        size="icon-sm"
        type="button"
        variant="ghost"
        data-testid="message-feedback-down"
        aria-label="Thumbs down"
        aria-pressed={feedback?.rating === -1}
        disabled={isSaving}
        className={cn(feedback?.rating === -1 && "text-primary bg-primary/10 hover:bg-primary/15")}
        onClick={() => void handleRate(-1)}
      >
        <ThumbsDownIcon size={12} />
      </Button>
    </>
  );
}

/**
 * Custom image component that handles artifact URLs
 */
function MessageImage({
  src,
  alt,
  threadId,
  maxWidth = "90%",
  ...props
}: React.ImgHTMLAttributes<HTMLImageElement> & {
  threadId: string;
  maxWidth?: string;
}) {
  if (!src) return null;

  const imgClassName = cn("overflow-hidden rounded-lg", `max-w-[${maxWidth}]`);

  if (typeof src !== "string") {
    return <img className={imgClassName} src={src} alt={alt} {...props} />;
  }

  const url = src.startsWith("/mnt/") ? resolveArtifactURL(src, threadId) : src;

  return (
    <a href={url} target="_blank" rel="noopener noreferrer">
      <img className={imgClassName} src={url} alt={alt} {...props} />
    </a>
  );
}

function MessageContent_({
  className,
  message,
  isLoading = false,
}: {
  className?: string;
  message: Message;
  isLoading?: boolean;
}) {
  const rehypePlugins = useRehypeSplitWordsIntoSpans(isLoading);
  const isHuman = message.type === "human";
  const { thread_id } = useParams<{ thread_id: string }>();
  const components = useMemo(
    () => ({
      img: (props: ImgHTMLAttributes<HTMLImageElement>) => (
        <MessageImage {...props} threadId={thread_id} maxWidth="90%" />
      ),
    }),
    [thread_id],
  );

  const rawContent = extractContentFromMessage(message);
  const reasoningContent = extractReasoningContentFromMessage(message);

  const files = useMemo(() => {
    const files = message.additional_kwargs?.files;
    if (!Array.isArray(files) || files.length === 0) {
      if (rawContent.includes("<uploaded_files>")) {
        // If the content contains the <uploaded_files> tag, we return the parsed files from the content for backward compatibility.
        return parseUploadedFiles(rawContent);
      }
      return null;
    }
    return files as FileInMessage[];
  }, [message.additional_kwargs?.files, rawContent]);

  const contentToDisplay = useMemo(() => {
    if (isHuman) {
      return rawContent ? stripUploadedFilesTag(rawContent) : "";
    }
    return rawContent ?? "";
  }, [rawContent, isHuman]);

  const filesList =
    files && files.length > 0 && thread_id ? (
      <RichFilesList files={files} threadId={thread_id} />
    ) : null;

  // Optimistic thinking state: show Reasoning block with elapsed timer
  if (message.additional_kwargs?._thinking) {
    return (
      <AIElementMessageContent className={className}>
        <Reasoning isStreaming={true}>
          <ReasoningTrigger />
          <ReasoningContent>{""}</ReasoningContent>
        </Reasoning>
      </AIElementMessageContent>
    );
  }

  // Uploading state: mock AI message shown while files upload
  if (message.additional_kwargs?.element === "task") {
    return (
      <AIElementMessageContent className={className}>
        <Task defaultOpen={false}>
          <TaskTrigger title="">
            <div className="text-muted-foreground flex w-full cursor-default items-center gap-2 text-sm select-none">
              <Loader className="size-4" />
              <span>{contentToDisplay}</span>
            </div>
          </TaskTrigger>
        </Task>
      </AIElementMessageContent>
    );
  }

  // Reasoning-only AI message (no main response content yet)
  if (!isHuman && reasoningContent && !rawContent) {
    return (
      <AIElementMessageContent className={className}>
        <Reasoning isStreaming={isLoading}>
          <ReasoningTrigger />
          <ReasoningContent>{reasoningContent}</ReasoningContent>
        </Reasoning>
      </AIElementMessageContent>
    );
  }

  if (isHuman) {
    const messageResponse = contentToDisplay ? (
      <AIElementMessageResponse
        remarkPlugins={humanMessagePlugins.remarkPlugins}
        rehypePlugins={humanMessagePlugins.rehypePlugins}
        components={components}
      >
        {contentToDisplay}
      </AIElementMessageResponse>
    ) : null;
    return (
      <div className={cn("ml-auto flex flex-col gap-2", className)}>
        {filesList}
        {messageResponse && (
          <AIElementMessageContent className="w-fit">
            {messageResponse}
          </AIElementMessageContent>
        )}
      </div>
    );
  }

  return (
    <AIElementMessageContent className={className}>
      {filesList}
      <MarkdownContent
        content={contentToDisplay}
        isLoading={isLoading}
        rehypePlugins={rehypePlugins}
        className="my-3"
        components={components}
      />
    </AIElementMessageContent>
  );
}

/**
 * Get file extension and check helpers
 */
const getFileExt = (filename: string) =>
  filename.split(".").pop()?.toLowerCase() ?? "";

const FILE_TYPE_MAP: Record<string, string> = {
  json: "JSON",
  csv: "CSV",
  txt: "TXT",
  md: "Markdown",
  py: "Python",
  js: "JavaScript",
  ts: "TypeScript",
  tsx: "TSX",
  jsx: "JSX",
  html: "HTML",
  css: "CSS",
  xml: "XML",
  yaml: "YAML",
  yml: "YAML",
  pdf: "PDF",
  png: "PNG",
  jpg: "JPG",
  jpeg: "JPEG",
  gif: "GIF",
  svg: "SVG",
  zip: "ZIP",
  tar: "TAR",
  gz: "GZ",
};

const IMAGE_EXTENSIONS = ["png", "jpg", "jpeg", "gif", "webp", "svg", "bmp"];

function getFileTypeLabel(filename: string): string {
  const ext = getFileExt(filename);
  return FILE_TYPE_MAP[ext] ?? (ext.toUpperCase() || "FILE");
}

function isImageFile(filename: string): boolean {
  return IMAGE_EXTENSIONS.includes(getFileExt(filename));
}

/**
 * Format bytes to human-readable size string
 */
function formatBytes(bytes: number): string {
  if (bytes === 0) return "—";
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  return `${(kb / 1024).toFixed(1)} MB`;
}

/**
 * List of files from additional_kwargs.files (with optional upload status)
 */
function RichFilesList({
  files,
  threadId,
}: {
  files: FileInMessage[];
  threadId: string;
}) {
  if (files.length === 0) return null;
  return (
    <div className="mb-2 flex flex-wrap justify-end gap-2">
      {files.map((file, index) => (
        <RichFileCard
          key={`${file.filename}-${index}`}
          file={file}
          threadId={threadId}
        />
      ))}
    </div>
  );
}

/**
 * Single file card that handles FileInMessage (supports uploading state)
 */
function RichFileCard({
  file,
  threadId,
}: {
  file: FileInMessage;
  threadId: string;
}) {
  const { t } = useI18n();
  const isUploading = file.status === "uploading";
  const isImage = isImageFile(file.filename);

  if (isUploading) {
    return (
      <div className="bg-background border-border/40 flex max-w-50 min-w-30 flex-col gap-1 rounded-lg border p-3 opacity-60 shadow-sm">
        <div className="flex items-start gap-2">
          <Loader2Icon className="text-muted-foreground mt-0.5 size-4 shrink-0 animate-spin" />
          <span
            className="text-foreground truncate text-sm font-medium"
            title={file.filename}
          >
            {file.filename}
          </span>
        </div>
        <div className="flex items-center justify-between gap-2">
          <Badge
            variant="secondary"
            className="rounded px-1.5 py-0.5 text-[10px] font-normal"
          >
            {getFileTypeLabel(file.filename)}
          </Badge>
          <span className="text-muted-foreground text-[10px]">
            {t.uploads.uploading}
          </span>
        </div>
      </div>
    );
  }

  if (!file.path) return null;

  const fileUrl = resolveArtifactURL(file.path, threadId);

  if (isImage) {
    return (
      <a
        href={fileUrl}
        target="_blank"
        rel="noopener noreferrer"
        className="group border-border/40 relative block overflow-hidden rounded-lg border"
      >
        <img
          src={fileUrl}
          alt={file.filename}
          className="h-32 w-auto max-w-60 object-cover transition-transform group-hover:scale-105"
        />
      </a>
    );
  }

  return (
    <div className="bg-background border-border/40 flex max-w-50 min-w-30 flex-col gap-1 rounded-lg border p-3 shadow-sm">
      <div className="flex items-start gap-2">
        <FileIcon className="text-muted-foreground mt-0.5 size-4 shrink-0" />
        <span
          className="text-foreground truncate text-sm font-medium"
          title={file.filename}
        >
          {file.filename}
        </span>
      </div>
      <div className="flex items-center justify-between gap-2">
        <Badge
          variant="secondary"
          className="rounded px-1.5 py-0.5 text-[10px] font-normal"
        >
          {getFileTypeLabel(file.filename)}
        </Badge>
        <span className="text-muted-foreground text-[10px]">
          {formatBytes(file.size)}
        </span>
      </div>
    </div>
  );
}

const MessageContent = memo(MessageContent_);
