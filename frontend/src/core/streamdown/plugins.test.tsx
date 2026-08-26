import { describe, expect, it } from "vitest";

import {
  humanMessagePlugins,
  rehypeSanitizeStep,
  remarkHumanHtmlAsText,
  streamdownPluginsWithWordAnimation,
} from "./plugins";

type TestNode = {
  type: string;
  value?: string;
  children?: TestNode[];
};

function collectNodeTypes(node: TestNode): string[] {
  return [
    node.type,
    ...(node.children ?? []).flatMap((child) => collectNodeTypes(child)),
  ];
}

describe("human message Streamdown plugins", () => {
  it("turns raw HTML markdown nodes into text nodes", () => {
    const tree = {
      type: "root",
      children: [
        { type: "html", value: "<dt>" },
        {
          type: "paragraph",
          children: [
            { type: "text", value: "prefix " },
            { type: "html", value: "<div>" },
          ],
        },
      ],
    };

    remarkHumanHtmlAsText()(tree);

    expect(tree.children[0]).toMatchObject({
      type: "text",
      value: "<dt>",
    });
    expect(tree.children[1]?.children?.[1]).toMatchObject({
      type: "text",
      value: "<div>",
    });
  });

  it("removes raw HTML wrappers while preserving fenced code nodes", () => {
    const tree: TestNode = {
      type: "root",
      children: [
        { type: "html", value: "<dt>" },
        { type: "code", value: "foundation" },
        { type: "html", value: "</dt>" },
      ],
    };

    remarkHumanHtmlAsText()(tree);

    expect(collectNodeTypes(tree)).not.toContain("html");
    expect(tree.children?.[0]).toMatchObject({
      type: "text",
      value: "<dt>",
    });
    expect(tree.children?.[1]).toMatchObject({
      type: "code",
      value: "foundation",
    });
    expect(tree.children?.[2]).toMatchObject({
      type: "text",
      value: "</dt>",
    });
  });

  it("registers the raw HTML textifier before math handling for human messages", () => {
    expect(humanMessagePlugins.remarkPlugins?.[0]).toBe(remarkHumanHtmlAsText);
  });

  it("keeps sanitization in custom rehype chains", () => {
    expect(streamdownPluginsWithWordAnimation.rehypePlugins).toContain(
      rehypeSanitizeStep,
    );
    expect(humanMessagePlugins.rehypePlugins).toContain(rehypeSanitizeStep);
  });
});
