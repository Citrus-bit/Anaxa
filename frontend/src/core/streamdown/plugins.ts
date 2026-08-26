import { defaultSchema } from "hast-util-sanitize";
import rehypeKatex from "rehype-katex";
import rehypeRaw from "rehype-raw";
import rehypeSanitize from "rehype-sanitize";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import type { StreamdownProps } from "streamdown";
import { visit } from "unist-util-visit";

import { rehypeSplitWordsIntoSpans } from "../rehype";

type RehypePlugin = NonNullable<StreamdownProps["rehypePlugins"]>[number];

const katexClassPattern = /^katex(?:-|$)/;
const mutedTextClassPattern = /^text-muted-foreground$/;

const sanitizedSchema = {
  ...defaultSchema,
  attributes: {
    ...defaultSchema.attributes,
    code: [
      ...(defaultSchema.attributes?.code ?? []),
      ["className", /^language-./],
    ],
    div: [
      ...(defaultSchema.attributes?.div ?? []),
      ["className", katexClassPattern],
    ],
    span: [
      ...(defaultSchema.attributes?.span ?? []),
      ["className", katexClassPattern],
      ["className", mutedTextClassPattern],
    ],
  },
  tagNames: [
    ...(defaultSchema.tagNames ?? []),
    "annotation",
    "math",
    "menclose",
    "mfrac",
    "mi",
    "mn",
    "mo",
    "mover",
    "mrow",
    "ms",
    "mspace",
    "msqrt",
    "mstyle",
    "msub",
    "msubsup",
    "msup",
    "mtable",
    "mtd",
    "mtext",
    "mtr",
    "munder",
    "munderover",
    "semantics",
  ],
};

export const rehypeSanitizeStep = [
  rehypeSanitize,
  sanitizedSchema,
] as RehypePlugin;

export const streamdownPlugins = {
  remarkPlugins: [
    remarkGfm,
    [remarkMath, { singleDollarTextMath: true }],
  ] as StreamdownProps["remarkPlugins"],
  rehypePlugins: [
    rehypeRaw,
    rehypeSanitizeStep,
    [rehypeKatex, { output: "html" }],
  ] as StreamdownProps["rehypePlugins"],
};

export const streamdownPluginsWithWordAnimation = {
  remarkPlugins: [
    remarkGfm,
    [remarkMath, { singleDollarTextMath: true }],
  ] as StreamdownProps["remarkPlugins"],
  rehypePlugins: [
    rehypeRaw,
    rehypeSanitizeStep,
    [rehypeKatex, { output: "html" }],
    rehypeSplitWordsIntoSpans,
  ] as StreamdownProps["rehypePlugins"],
};

type MarkdownNode = {
  type: string;
  children?: MarkdownNode[];
  value?: string;
};

type MarkdownHtmlNode = MarkdownNode & {
  type: "html" | "text";
  value?: string;
};

export function remarkHumanHtmlAsText() {
  return (tree: MarkdownNode) => {
    visit(tree, "html", (node: MarkdownHtmlNode) => {
      node.type = "text";
    });
  };
}

// Plugins for human messages - no autolink to prevent URL bleeding into adjacent text
export const humanMessagePlugins = {
  remarkPlugins: [
    remarkHumanHtmlAsText,
    // Use remark-gfm without autolink literals by not including it
    // Only include math support for human messages
    [remarkMath, { singleDollarTextMath: true }],
  ] as StreamdownProps["remarkPlugins"],
  rehypePlugins: [
    rehypeSanitizeStep,
    [rehypeKatex, { output: "html" }],
  ] as StreamdownProps["rehypePlugins"],
};
