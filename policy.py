from enum import Enum, auto
from pacsltk import perfmodel

import conf

COLD_START_PROB_INITIAL_GUESS = 0.0

class SchedulerDecision(Enum):
    EXEC = 1
    OFFLOAD_CLOUD = 2
    OFFLOAD_EDGE = 3
    DROP = 4

class ColdStartEstimation(Enum):
    NO = auto()
    NAIVE = auto()
    NAIVE_PER_FUNCTION = auto()
    PACS = auto()
    FULL_KNOWLEDGE = auto()

    @classmethod
    def from_string(cls,s):
        s = s.lower()
        if s == "no":
            return ColdStartEstimation.NO
        elif s == "naive" or s == "":
            return ColdStartEstimation.NAIVE
        elif s == "naive-per-function":
            return ColdStartEstimation.NAIVE_PER_FUNCTION
        elif s == "pacs":
            return ColdStartEstimation.PACS
        elif s == "full-knowledge":
            return ColdStartEstimation.FULL_KNOWLEDGE
        return None


class Policy:

    def __init__(self, simulation, node):
        self.simulation = simulation
        self.node = node
        self.__edge_peers = None
        self.budget = simulation.config.getfloat(conf.SEC_POLICY, conf.HOURLY_BUDGET, fallback=-1.0)
        self.local_budget = self.budget
        if simulation.config.getboolean(conf.SEC_POLICY, conf.SPLIT_BUDGET_AMONG_EDGE_NODES, fallback=False):
            nodes = len(simulation.infra.get_edge_nodes())
            self.local_budget = self.budget / nodes


    def schedule(self, function, qos_class, offloaded_from):
        pass

    def update(self):
        pass

    def can_execute_locally(self, f, reclaim_memory=True):
        if f in self.node.warm_pool or self.node.curr_memory >= f.memory:
            return True
        if reclaim_memory:
            reclaimed = self.node.warm_pool.reclaim_memory(f.memory - self.node.curr_memory)
            self.node.curr_memory += reclaimed
        return self.node.curr_memory >= f.memory

    def _get_edge_peers (self):
        if self.__edge_peers is None:
            # TODO: need to refresh over time?
            self.__edge_peers = self.simulation.infra.get_neighbors(self.node, self.simulation.node_choice_rng, self.simulation.max_neighbors)
        return self.__edge_peers

    def _get_edge_peers_probabilities (self):
        peers = self._get_edge_peers()
            #assert(peer.curr_memory*peer.peer_exposed_memory_fraction >= 0.0)

        total_memory = sum([max(0,x.curr_memory)*x.peer_exposed_memory_fraction for x in peers])
        if total_memory > 0.0:
            probs = [max(0,x.curr_memory)*x.peer_exposed_memory_fraction/total_memory for x in peers]
        else:
            n = len(peers)
            probs = [1.0/n for x in peers]
        return probs, peers

    # Picks a node for Edge offloading
    def pick_edge_node (self, fun, qos):
        # Pick peers based on resource availability
        probs, peers = self._get_edge_peers_probabilities()
        if len(peers) < 1:
            return None
        return self.simulation.node_choice_rng.choice(peers, p=probs)



class BasicPolicy(Policy):

    def schedule(self, f, c, offloaded_from):
        if self.can_execute_locally(f):
            return (SchedulerDecision.EXEC, None)
        else:
            return (SchedulerDecision.OFFLOAD_CLOUD, None)

class BasicBudgetAwarePolicy(Policy):

    def schedule(self, f, c, offloaded_from):
        budget_ok = self.simulation.stats.cost / self.simulation.t * 3600 < self.budget

        if self.can_execute_locally(f):
            return (SchedulerDecision.EXEC, None)
        elif self.simulation.stats.cost / self.simulation.t * 3600 < self.budget:
            return (SchedulerDecision.OFFLOAD_CLOUD, None)
        else:
            return (SchedulerDecision.DROP, None)

class BasicEdgePolicy(Policy):

    def schedule(self, f, c, offloaded_from):
        if self.can_execute_locally(f):
            return (SchedulerDecision.EXEC, None)
        elif len(offloaded_from) == 0:
            return (SchedulerDecision.OFFLOAD_EDGE, self.pick_edge_node(f,c))
        else:
            return (SchedulerDecision.DROP, None)

class CloudPolicy(Policy):

    def schedule(self, f, c, offloaded_from):
        if self.can_execute_locally(f):
            return (SchedulerDecision.EXEC, None)
        else:
            return (SchedulerDecision.DROP, None)



class GreedyPolicy(Policy):

    def __init__(self, simulation, node):
        super().__init__(simulation, node)
        self.cold_start_prob = {}
        self.cold_start_prob_cloud = {}
        self.estimated_service_time = {}
        self.estimated_service_time_cloud = {}

        self.local_cold_start_estimation = ColdStartEstimation.from_string(self.simulation.config.get(conf.SEC_POLICY, conf.LOCAL_COLD_START_EST_STRATEGY, fallback=""))
        self.cloud_cold_start_estimation = ColdStartEstimation.from_string(self.simulation.config.get(conf.SEC_POLICY, conf.CLOUD_COLD_START_EST_STRATEGY, fallback=""))

        # OLD: cloud_region = node.region.default_cloud
        #self.cloud = self.simulation.node_choice_rng.choice(self.simulation.infra.get_region_nodes(cloud_region), 1)[0]

        # Pick the closest cloud node
        nodes_w_lat = [(_n,simulation.infra.get_latency(node,_n)) for _n in simulation.infra.get_cloud_nodes()]
        self.cloud = sorted(nodes_w_lat, key=lambda x: x[1])[0][0]

    def _estimate_latency (self, f, c):
        if self.local_cold_start_estimation == ColdStartEstimation.FULL_KNOWLEDGE:
            if f in self.node.warm_pool:
                self.cold_start_prob[(f, self.node)] = 0
            else:
                self.cold_start_prob[(f, self.node)] = 1
        if self.cloud_cold_start_estimation == ColdStartEstimation.FULL_KNOWLEDGE:
            if f in self.cloud.warm_pool:
                self.cold_start_prob[(f, self.cloud)] = 0
            else:
                self.cold_start_prob[(f, self.cloud)] = 1

        latency_local = self.estimated_service_time.get(f, 0) + \
                        self.cold_start_prob.get((f, self.node), 1) * \
                        self.simulation.init_time[(f,self.node)]

        latency_cloud = self.estimated_service_time_cloud.get(f, 0) +\
                2 * self.simulation.infra.get_latency(self.node, self.cloud) + \
                        self.cold_start_prob.get((f, self.cloud), 1) * self.simulation.init_time[(f,self.cloud)] +\
                        f.inputSizeMean*8/1000/1000/self.simulation.infra.get_bandwidth(self.node, self.cloud)
        return (latency_local, latency_cloud)

    def schedule(self, f, c, offloaded_from):
        latency_local, latency_cloud = self._estimate_latency(f,c)

        if self.can_execute_locally(f) and latency_local < latency_cloud:
            return (SchedulerDecision.EXEC, None)
        else:
            return (SchedulerDecision.OFFLOAD_CLOUD, self.cloud)

    def update_cold_start (self, stats):
        #
        # LOCAL NODE
        #
        if self.local_cold_start_estimation == ColdStartEstimation.PACS:
            for f in self.simulation.functions:
                total_arrival_rate = max(0.001, sum([stats.arrivals.get((f,x,self.node), 0.0) for x in self.simulation.classes])/self.simulation.t)
                props1, _ = perfmodel.get_sls_warm_count_dist(total_arrival_rate,
                                                            self.estimated_service_time[f],
                                                            self.estimated_service_time[f] + self.simulation.init_time[(f,self.node)],
                                                            self.simulation.expiration_timeout)
                self.cold_start_prob[(f, self.node)] = props1["cold_prob"]
        elif self.local_cold_start_estimation == ColdStartEstimation.NAIVE:
            # Same prob for every function
            node_compl = sum([stats.node2completions[(_f,self.node)] for _f in self.simulation.functions])
            node_cs = sum([stats.cold_starts[(_f,self.node)] for _f in self.simulation.functions])
            for f in self.simulation.functions:
                if node_compl > 0:
                    self.cold_start_prob[(f, self.node)] = node_cs / node_compl
                else:
                    self.cold_start_prob[(f, self.node)] = COLD_START_PROB_INITIAL_GUESS
        elif self.local_cold_start_estimation == ColdStartEstimation.NAIVE_PER_FUNCTION:
            for f in self.simulation.functions:
                if stats.node2completions.get((f,self.node), 0) > 0:
                    self.cold_start_prob[(f, self.node)] = stats.cold_starts.get((f,self.node),0) / stats.node2completions.get((f,self.node),0)
                else:
                    self.cold_start_prob[(f, self.node)] = COLD_START_PROB_INITIAL_GUESS
        elif self.local_cold_start_estimation == ColdStartEstimation.NO: 
            for f in self.simulation.functions:
                self.cold_start_prob[(f, self.node)] = 0

        # CLOUD
        #
        if self.cloud_cold_start_estimation == ColdStartEstimation.PACS:
            for f in self.simulation.functions:
                total_arrival_rate = max(0.001, sum([stats.arrivals.get((f,x,self.cloud), 0.0) for x in self.simulation.classes])/self.simulation.t)
                props1, _ = perfmodel.get_sls_warm_count_dist(total_arrival_rate,
                                                            self.estimated_service_time[f],
                                                            self.estimated_service_time[f] + self.simulation.init_time[(f,self.cloud)],
                                                            self.simulation.expiration_timeout)
                self.cold_start_prob[(f, self.cloud)] = props1["cold_prob"]
        elif self.cloud_cold_start_estimation == ColdStartEstimation.NAIVE:
            # Same prob for every function
            node_compl = sum([stats.node2completions[(_f,self.cloud)] for _f in self.simulation.functions])
            node_cs = sum([stats.cold_starts[(_f,self.cloud)] for _f in self.simulation.functions])
            for f in self.simulation.functions:
                if node_compl > 0:
                    self.cold_start_prob[(f, self.cloud)] = node_cs / node_compl
                else:
                    self.cold_start_prob[(f, self.cloud)] = COLD_START_PROB_INITIAL_GUESS
        elif self.cloud_cold_start_estimation == ColdStartEstimation.NAIVE_PER_FUNCTION:
            for f in self.simulation.functions:
                if stats.node2completions.get((f,self.cloud), 0) > 0:
                    self.cold_start_prob[(f, self.cloud)] = stats.cold_starts.get((f,self.cloud),0) / stats.node2completions.get((f,self.cloud),0)
                else:
                    self.cold_start_prob[(f, self.cloud)] = COLD_START_PROB_INITIAL_GUESS
        elif self.cloud_cold_start_estimation == ColdStartEstimation.NO: 
            for f in self.simulation.functions:
                self.cold_start_prob[(f, self.cloud)] = 0

    def update(self):
        stats = self.simulation.stats

        for f in self.simulation.functions:
            if stats.node2completions[(f, self.node)] > 0:
                self.estimated_service_time[f] = stats.execution_time_sum[(f, self.node)] / \
                                                 stats.node2completions[(f, self.node)]
            else:
                self.estimated_service_time[f] = 0.1
            if stats.node2completions[(f, self.cloud)] > 0:
                self.estimated_service_time_cloud[f] = stats.execution_time_sum[(f, self.cloud)] / \
                                                       stats.node2completions[(f, self.cloud)]
            else:
                self.estimated_service_time_cloud[f] = 0.1

        self.update_cold_start(stats)

class GreedyBudgetAware(GreedyPolicy):

    def __init__ (self, simulation, node):
        super().__init__(simulation, node)

    def schedule(self, f, c, offloaded_from):
        latency_local, latency_cloud = self._estimate_latency(f,c)
        local_ok = self.can_execute_locally(f)
        budget_ok = self.simulation.stats.cost / self.simulation.t * 3600 < self.budget

        if not budget_ok and not local_ok:
            return (SchedulerDecision.DROP, None)
        if local_ok and latency_local < latency_cloud:
            return (SchedulerDecision.EXEC, None)
        if budget_ok:
            return (SchedulerDecision.OFFLOAD_CLOUD, self.cloud)
        else:
            return (SchedulerDecision.EXEC, None)


class GreedyPolicyWithCostMinimization(GreedyPolicy):

    def __init__ (self, simulation, node):
        super().__init__(simulation, node)
        # Pick the closest cloud node
        nodes_w_lat = [(_n,simulation.infra.get_latency(node,_n)) for _n in simulation.infra.get_cloud_nodes()]
        self.cloud = sorted(nodes_w_lat, key=lambda x: x[1])[0][0]

    def schedule(self, f, c, offloaded_from):
        if self.local_cold_start_estimation == ColdStartEstimation.FULL_KNOWLEDGE:
            if f in self.node.warm_pool:
                self.cold_start_prob[(f, self.node)] = 0
            else:
                self.cold_start_prob[(f, self.node)] = 1
        if self.cloud_cold_start_estimation == ColdStartEstimation.FULL_KNOWLEDGE:
            if f in self.cloud.warm_pool:
                self.cold_start_prob[(f, self.cloud)] = 0
            else:
                self.cold_start_prob[(f, self.cloud)] = 1

        latency_local = self.estimated_service_time.get(f, 0) + \
                        self.cold_start_prob.get((f, self.node), 1) * \
                        self.simulation.init_time[(f,self.node)]

        latency_cloud = self.estimated_service_time_cloud.get(f, 0) + 2 * self.simulation.infra.get_latency(self.node, self.cloud) + \
                        self.cold_start_prob.get((f, self.cloud), 1) * self.simulation.init_time[(f,self.cloud)] +\
                        f.inputSizeMean*8/1000/1000/self.simulation.infra.get_bandwidth(self.node, self.cloud)

        if latency_local < c.max_rt and self.can_execute_locally(f):
            # Choose the configuration with minimum cost (edge execution) if both configuration can execute within
            # the deadline
            sched_decision = SchedulerDecision.EXEC, None
        elif latency_cloud < c.max_rt:
            sched_decision = SchedulerDecision.OFFLOAD_CLOUD, self.cloud
        elif self.can_execute_locally(f):
            sched_decision = SchedulerDecision.EXEC, None
        else:
            sched_decision = SchedulerDecision.OFFLOAD_CLOUD, self.cloud

        return sched_decision

