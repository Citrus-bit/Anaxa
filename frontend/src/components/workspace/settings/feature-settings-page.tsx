"use client";

import { BotIcon, PuzzleIcon, ServerIcon } from "lucide-react";
import type { ReactNode } from "react";

import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { useFeatures } from "@/core/features";
import type {
  FeatureAgent,
  FeatureSkill,
  FeatureTool,
  RedactedConfigKey,
} from "@/core/features";
import { useI18n } from "@/core/i18n/hooks";

import { SettingsErrorState } from "./settings-error-state";
import { SettingsSection } from "./settings-section";

export function FeatureSettingsPage() {
  const { t } = useI18n();
  const { features, isLoading, error, refetch } = useFeatures();

  if (isLoading) {
    return (
      <SettingsSection
        title={t.settings.features.title}
        description={t.settings.features.description}
      >
        <div className="grid gap-4 lg:grid-cols-3">
          {[1, 2, 3].map((item) => (
            <Skeleton key={item} className="h-48 rounded-lg" />
          ))}
        </div>
      </SettingsSection>
    );
  }

  if (error) {
    return (
      <SettingsSection
        title={t.settings.features.title}
        description={t.settings.features.description}
      >
        <SettingsErrorState error={error} onRetry={() => void refetch()} />
      </SettingsSection>
    );
  }

  return (
    <div className="space-y-8">
      <SettingsSection
        title={t.settings.features.title}
        description={t.settings.features.description}
      >
        <div className="text-muted-foreground text-sm">
          {t.settings.features.readonlyHint}
        </div>
      </SettingsSection>

      <FeatureGroup
        icon={<BotIcon className="size-4" />}
        title={t.settings.features.agentsTitle}
        count={features.agents.length}
        empty={t.settings.features.emptyAgents}
      >
        {features.agents.map((agent) => (
          <AgentFeatureCard key={agent.name} agent={agent} />
        ))}
      </FeatureGroup>

      <FeatureGroup
        icon={<ServerIcon className="size-4" />}
        title={t.settings.features.toolsTitle}
        count={features.tools.length}
        empty={t.settings.features.emptyTools}
      >
        {features.tools.map((tool) => (
          <ToolFeatureCard key={tool.name} tool={tool} />
        ))}
      </FeatureGroup>

      <FeatureGroup
        icon={<PuzzleIcon className="size-4" />}
        title={t.settings.features.skillsTitle}
        count={features.skills.length}
        empty={t.settings.features.emptySkills}
      >
        {features.skills.map((skill) => (
          <SkillFeatureCard key={`${skill.category}:${skill.name}`} skill={skill} />
        ))}
      </FeatureGroup>
    </div>
  );
}

function FeatureGroup({
  icon,
  title,
  count,
  empty,
  children,
}: {
  icon: ReactNode;
  title: string;
  count: number;
  empty: string;
  children: ReactNode;
}) {
  return (
    <section className="space-y-3">
      <div className="flex items-center gap-2">
        <div className="bg-muted flex size-8 items-center justify-center rounded-md">
          {icon}
        </div>
        <h3 className="text-sm font-semibold">{title}</h3>
        <Badge variant="secondary">{count}</Badge>
      </div>
      {count === 0 ? (
        <div className="text-muted-foreground rounded-lg border border-dashed p-4 text-sm">
          {empty}
        </div>
      ) : (
        <div className="grid gap-3 lg:grid-cols-2">{children}</div>
      )}
    </section>
  );
}

function AgentFeatureCard({ agent }: { agent: FeatureAgent }) {
  const { t } = useI18n();
  return (
    <div className="space-y-3 rounded-lg border p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="truncate text-sm font-semibold">{agent.name}</div>
          <p className="text-muted-foreground mt-1 line-clamp-3 text-xs">
            {agent.description ?? t.settings.features.noDescription}
          </p>
        </div>
        <Badge variant={agent.kind === "system" ? "default" : "secondary"}>
          {agent.kind}
        </Badge>
      </div>
      <div className="flex flex-wrap gap-1.5">
        <Badge variant="outline">
          {agent.readonly
            ? t.settings.features.readonly
            : t.settings.features.customEditable}
        </Badge>
        {agent.model && <Badge variant="outline">{agent.model}</Badge>}
        {(agent.tool_groups ?? []).map((group) => (
          <Badge key={group} variant="outline">
            {group}
          </Badge>
        ))}
      </div>
    </div>
  );
}

function ToolFeatureCard({ tool }: { tool: FeatureTool }) {
  const { t } = useI18n();
  return (
    <div className="space-y-3 rounded-lg border p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="truncate text-sm font-semibold">{tool.name}</div>
          <p className="text-muted-foreground mt-1 line-clamp-3 text-xs">
            {tool.description ?? t.settings.features.noDescription}
          </p>
        </div>
        <Badge variant={tool.enabled ? "default" : "outline"}>
          {tool.enabled ? t.settings.features.enabled : t.settings.features.disabled}
        </Badge>
      </div>
      <dl className="grid gap-2 text-xs">
        <FeatureRow label={t.settings.features.transport} value={tool.transport} />
        <FeatureRow
          label={t.settings.features.endpoint}
          value={tool.url ?? tool.command ?? t.settings.features.notConfigured}
        />
        {tool.args.length > 0 && (
          <FeatureRow label={t.settings.features.arguments} value={tool.args.join(" ")} />
        )}
      </dl>
      <RedactedKeyList label={t.settings.features.envKeys} items={tool.env_keys} />
      <RedactedKeyList
        label={t.settings.features.headerKeys}
        items={tool.header_keys}
      />
      {tool.oauth_enabled && (
        <Badge variant="outline">{t.settings.features.oauthEnabled}</Badge>
      )}
    </div>
  );
}

function SkillFeatureCard({ skill }: { skill: FeatureSkill }) {
  const { t } = useI18n();
  return (
    <div className="space-y-3 rounded-lg border p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="truncate text-sm font-semibold">{skill.name}</div>
          <p className="text-muted-foreground mt-1 line-clamp-3 text-xs">
            {skill.description ?? t.settings.features.noDescription}
          </p>
        </div>
        <Badge variant={skill.enabled ? "default" : "outline"}>
          {skill.enabled ? t.settings.features.enabled : t.settings.features.disabled}
        </Badge>
      </div>
      <div className="flex flex-wrap gap-1.5">
        <Badge variant="secondary">{skill.category}</Badge>
        {skill.license && <Badge variant="outline">{skill.license}</Badge>}
      </div>
    </div>
  );
}

function FeatureRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="grid grid-cols-[96px_1fr] gap-2">
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="truncate font-mono">{value}</dd>
    </div>
  );
}

function RedactedKeyList({
  label,
  items,
}: {
  label: string;
  items: RedactedConfigKey[];
}) {
  if (items.length === 0) return null;
  return (
    <div className="space-y-1">
      <div className="text-muted-foreground text-xs">{label}</div>
      <div className="flex flex-wrap gap-1.5">
        {items.map((item) => (
          <Badge
            key={item.key}
            variant={item.configured ? "outline" : "secondary"}
            className="font-mono"
          >
            {item.key}
          </Badge>
        ))}
      </div>
    </div>
  );
}
