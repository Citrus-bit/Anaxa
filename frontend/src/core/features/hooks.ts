import { useQuery } from "@tanstack/react-query";

import {
  fetchBrowserControlEnabled,
  fetchMcpTasksEnabled,
  fetchSubagentBatchesCapability,
  loadFeatures,
} from "./api";

export const featuresQueryOptions = {
  queryKey: ["features", "inventory"],
  queryFn: () => loadFeatures(),
  staleTime: 30_000,
  retry: 2,
  retryDelay: 1_000,
} as const;

export function useFeatures() {
  const { data, isLoading, error, refetch } = useQuery(featuresQueryOptions);
  return {
    features: data ?? { agents: [], tools: [], skills: [] },
    isLoading,
    error,
    refetch,
  };
}

export function useBrowserControlEnabled() {
  const { data, isPending } = useQuery({
    queryKey: ["features", "browser_control"],
    queryFn: () => fetchBrowserControlEnabled(),
    staleTime: 0,
    refetchOnMount: true,
    retry: false,
  });

  return {
    enabled: data ?? false,
    isLoading: isPending,
  };
}

export function useMcpTasksEnabled() {
  const { data, isPending } = useQuery({
    queryKey: ["features", "mcp_tasks"],
    queryFn: () => fetchMcpTasksEnabled(),
    staleTime: 0,
    refetchOnMount: true,
    retry: false,
  });

  return {
    enabled: data ?? false,
    isLoading: isPending,
  };
}

export function useSubagentBatchesCapability() {
  const { data, isPending } = useQuery({
    queryKey: ["features", "subagent_batches"],
    queryFn: () => fetchSubagentBatchesCapability(),
    staleTime: 0,
    refetchOnMount: true,
    retry: false,
  });
  return {
    repositoryAvailable: data?.repositoryAvailable ?? false,
    workerRunning: data?.workerRunning ?? false,
    maxRunning: data?.maxRunning ?? 0,
    isLoading: isPending,
  };
}
