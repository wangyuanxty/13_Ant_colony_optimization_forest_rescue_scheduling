# Scheduling and route planning for forests rescue: Applications with a novel ant colony optimization algorithm 

Xu Wangying, Xie Naiming 임<br>Show more<br>Outline<br>Share<br>Cite

https://doi.org/10.1016/j.engappai.2025.111042 π
Get rights and content $\lambda$

## Highlights

- For forest disaster prevention, helicopter inspection is vital. We propose a two-stage dynamic path planning model to handle forest environmental uncertainties and boost prevention efficiency.
- The two-stage model has offline and online scheduling. Offline sets daily helicopter patrol routes; online manages inspection - time abnormal situations.
- To overcome late-stage taboo search list limits, we introduce IACO. Its new pheromone method helps the algorithm avoid local optima and get optimal results.
- Two case studies are shown. One is a helicopter's response to a regional alarm, and the other is inter-regional joint dispatch, validating the model's practicality.


#### Abstract

With the increasing frequency and intensity of forest fires due to global climate change and human activities, timely and efficient fire rescue has become critical. Helicopter inspections are a preventive measure that significantly improves fire incident resolution efficiency. Existing approaches to helicopter route planning often focus on single-stage scheduling (either offline pre-planning or online reactive adjustments), which fail to dynamically balance pre-scheduled routes with real-time emergency responses. Additionally, traditional optimization algorithms like the Ant Colony Optimization (ACO) suffer from premature convergence and instability


caused by overly random initial search solutions and restrictive tabu lists in later search phases. To address these gaps, this paper proposes a two-stage dynamic path planning model that integrates offline pre-scheduling with online adaptive adjustments. The model first optimizes inspection routes for all points using a novel Improved Ant Colony Optimization (IACO) algorithm, which introduces dynamic pheromone initialization and a stage-dependent pheromone update strategy to enhance exploration-exploitation balance. During real-time operations, when emergencies occur, the model dynamically dispatches the nearest available helicopter while reconfiguring remaining inspection paths. Two representative scenarios validate the model's effectiveness: (1) independent regional dispatch where a helicopter adjusts its route to handle an emergency within its assigned cluster, and (2) cross-region joint dispatch involving multi-helicopter coordination across clusters. The IACO algorithm outperforms traditional ACO by reducing total flight distances by up to $4.1 \%$ in offline planning and improving emergency response efficiency by $7.3 \%$ in dynamic scenarios. This work provides a robust framework for balancing routine inspections with agile emergency management in forest fire rescue operations.
□

## Keywords

Forestry inspection; Helicopter route planning; Ant colony algorithm; Dynamic dispatch

## 1. Introduction

Forest fires have dramatically increased in countries worldwide in the face of global warming and increased idiosyncratic weather (Randerson et al., 2006). Forest fires are a widespread problem worldwide. These problems have serious consequences. In November 2018, the Ulsey Hill Fire and the Camp Hill Fire in California resulted in 88 deaths and 249 missing, and the fires destroyed more than 20,000 buildings, with an estimated loss of more than $\$ 20$ billion (Safford et al., 2022). The California fire caused one of the most destructive forest fires in the history of the United States (Brown et al., 2020). Efficient utilization of high-tech resources is one of the important means of relief. However, due to the complexity of the natural environment, scientific rescue is not known in many cases. Helicopter inspection is a worthwhile and affordable mean of dealing with disasters. Earlier and faster detection of fires and timely reporting to take measures will significantly improve the efficiency and speed of rescue and reduce losses. Therefore, how to plan and design inspection paths is a crucial problem for disaster relief.

Forest aviation firefighting integrates high and new technologies. Scientists have conducted a lot of research on how to effectively improve rescue efficiency and minimize disaster losses (Ajith and Jolly, 2021). Afonso. et al. (E. Carvalho et al., 2024). proposed a new traversability analysis and path planning technique to generate effective paths by processing 3D point cloud maps to compute terrain gradient information and detecting the presence of obstacles. Wang et al. (2023a) proposed an AI-Based Action Detection UAV System, which improves firefighter performance by monitoring fire characteristics and inferring trajectories. A multi-robot system for GPS-denied search and rescue under the forest canopy was proposed by Tian et al. (2020). It utilizes UAV to transmit compressed tree-based sub-maps to a central ground station for collaborative simultaneous localization and mapping. Xu et al. (2021) proposed the High-Performance Emergency Rescue Management e-System (PERMS), which is an efficient rescue route planning scheme that operates within a high-performance vehicular emergency management system based on a mobile cloud computing paradigm. A growing interest in utilizing AI mechanisms for forest fire management. Researchers tend to use computer software to simulate fire scenarios and use AI methods combined with computers to help solve problems (Ebrahimi et al., 2021; Giannakidou et al., 2024; Guha et al., 2022; Jazebi et al., 2019; Kyrkou et al., 2022).

Helicopter rescue has the advantages of low dispatch cost, high safety, and long endurance and has significant advantages in daily inspection (Meadley et al., 2021). In order to ensure that the helicopter completes a specific task in an effective time, it is necessary to plan the shortest path according to the actual inspection point and flight
space (Li and Cheng, 2023; Zhao and Wang, 2024). Helicopter rescue flight scheduling and path planning are important influencing factors to ensure flight safety and improve rescue efficiency. The flight path of helicopters is a key element in the development and implementation of the entire rescue program, which directly affects the effectiveness of the entire rescue operation (Zhang et al., 2022). A significant amount of effort is expended in path planning. Zhang et al. (2020) proposed a rescue system consisting of a rescue simulator and a rescue algorithm characterized by supporting offline simulation of the dynamic rescue process between forest fire spread and forest fire rescue. Zhang et al. (2023) combines a heuristic crossover strategy with the basic SAR algorithm, and a realtime path adjustment strategy is added to a real-time path adjustment strategy. Michael et al. (Morin et al., 2023) developed an ant colony algorithm to discover the probability of a moving search object with Markovian motion, allocating visible regions over a limited time horizon and resource scanning process. Sun et al. (2022) gave a hierarchical-based pheromone update strategy and partition-based pheromone management mechanism to solve the drone scheduling problem (DSP). In summary, helicopters or drones, which are rescue tools that fly without regard to roadblocks, are the most preferred for inspection or rescue sessions (Al-Hilo et al., 2020; B et al., 2021; Wang et al., 2023b).

Based on the above-existing research results, scholars mostly tend to study single or multiple helicopter scheduling in a single phase currently. Some studies considered a partial approach in terms of daily inspections. They only consider preemptive offline scheduling before the helicopter departs, or only dynamic scheduling based on the helicopter's current posture, and therefore lack comprehensive and systematic planning for the flight process. Fewer scholars have combined offline scheduling with dynamic scheduling to establish scheduling models. The scientific and efficient scheduling of forest helicopter patrols can prevent the key aspects of the occurrence of emergencies, and is crucial for the resource scheduling of subsequent emergencies. Therefore, this paper establishes a two-stage helicopter group dynamic path planning model that considers tentative emergencies. In the first stage, the helicopters obtain the flight paths traversing each inspection point through offline scheduling. In the second stage, when an emergency occurs during helicopter inspections, the appropriate helicopter is dispatched to inspect the site based on its current location and state. The two-stage model improves the overall inspection efficiency of the helicopter. In addition, the IACO algorithm is proposed to solve the model more efficiently. Improvements are aimed at addressing the instability caused by the overly random initial search solution and the subsequent search phase constrained by the tabu search list.

The rest of the work is organized as follows. In Section 2, a two-stage helicopter dynamic inspection model is developed and a parametric and mathematical description of the problem is given. In Section 3, we describe an IACO algorithm for multiple helicopters and combine it with a K-means clustering algorithm to solve the proposed mathematical modeling problem. Some function test is also conducted in section 3 to examine the performance of IACO. Section 4 is the experimental section. We give two cases to describe the dynamic patrol routes of helicopters in different situations. Finally, conclusions are given and relevant outlooks are presented in Section 5.

## 2. Mathematic model

### 2.1. Selection of model in forest inspection

Traditional route planning problems can be addressed using either the Vehicle Routing Problem (VRP) model or the Traveling Salesman Problem (TSP) model (Bektas, 2006). While VRP theoretically optimizes routes for all points simultaneously, it requires consideration of complex constraints such as vehicle capacity and time windows, significantly increasing computational complexity. In the forest inspection scenario described in this article, helicopter endurance capacity is primarily constrained by battery power or fuel, unlike the dynamic and variable vehicle capacities in VRP. From a problem characteristics perspective, the flight patterns of helicopters in forest inspection scenarios are highly aligned with TSP requirements (Mestria and Engineering, 2018). The forest environment is complex, with inspection points widely distributed and relatively scattered, while TSP's focus on finding optimal paths to visit all points aligns closely with the practical requirement for helicopters to inspect each location sequentially.

In the context of forest fire prevention and rescue, "suppress fires at their earliest, smallest, and most manageable stages" represents a critical objective for achieving rapid response mechanisms. Therefore, this paper employs a clustering-based TSP approach, decomposing large-scale problems into multiple small-scale TSP subproblems. This strategy significantly reduces computational complexity while improving solution efficiency. When faced with large numbers of inspection points in later stages, this method can generate optimized routes in shorter timeframes, meeting the rapid decision-making requirements of forest fire emergency responses.

### 2.2. Forest inspection mathematic model

The entire tour is divided into two stages. Offline scheduling is the first stage. First, all the coordinates of the tour points are classified and combined with the clustering method. The exact number of classifications depends on the number of helicopters dispatched. The second stage is online scheduling. There is a high probability that emergencies may occur considering the uncertainty of the forest environment. The command center needs to dispatch the nearest helicopter to the inspection in time according to the time and location of the emergent situation. Table 1 shows the specific parameters of the mathematical model.

Table 1. Specific parameters of the mathematic model.
| Type | Symbols | Descriptions |
| :--- | :--- | :--- |
| Stage 1 | $U$ | Set of helicopters |
|  | $n$ | Number of helicopters |
|  | $h$ | The maximum flight distance of the helicopter |
|  | A | Set of prior access points |
|  | B | Set of improvised reports requiring visits |
|  | M | Set of all the inspection points, $M=A \cup B$ |
|  | $d_{i j}$ | Distance between $i$ and $j$ |
| Stage 2 | $S_{t}$ | The set of patrol points that have been traversed at $t$. The points in $S_{t}$ must belong to M |
|  | $l_{i j}$ | Off-line scheduling of helicopter tour routes. $i j$ denotes the connecting line between the $i$ th and $j$ th points. |
|  | $W$ | Set of inspection points in order of visit according to offline scheduling |
|  | $T$ | Collection of moments of alarm occurrence |
|  | $N$ | A number of alarms occurred. |
|  | $t$ | The current time of the alarm |
|  | $v$ | Helicopter traveling speed |
|  | $k$ | Serial number of the helicopter |
|  | $k_{\text {min }}$ | The current helicopter number closest to the alarm point |
|  | $P_{k}$ | Current location of the $k$ th helicopter from the alarm point |
|  | $Q_{t}$ | Collection of the last helicopter patrol point locations before an alarm occurs |
|  | $P_{m n}^{k}$ | The current position of the $k$ th helicopter at the time the alarm occurred |
|  | C | $C=\{0,1,2\}$, which denotes the 3 different states of the helicopters. |
| Decision variables | $x_{i j}^{k}$ | Decision variable. It is used to determine whether helicopter $k$ passes the route between points $i$ and $j$. $i, j$ is the inspection points number and $k$ is the helicopter number. |


Considering the simplicity and effectiveness of the model, the states of the helicopter are classified into three categories: normal patrol flight, flying to the alarm point for inspection, and conducting inspection at the alarm point. We are committed to constructing an efficient forest fire rescue helicopter scheduling and route planning model to improve the rescue efficiency, whose core lies in optimizing the route planning of helicopters during patrols and in response to emergencies. These three states are sufficient to clearly define the key stages of the helicopter during the task execution, and can reflect the key influencing factors on route planning and scheduling decisions in different stages. Fig. 1 shows the different state paths for helicopters traveling to alarm points and Table 2 shows the helicopter state codes and definitions.

![](https://cdn.mathpix.com/cropped/286e4446-4f36-48bc-9a51-5f151b677c98-05.jpg?height=490&width=810&top_left_y=519&top_left_x=155)
Fig. 1. Different states of helicopters on patrol.

Download: Download high-res image $(223 \mathrm{~KB})$
Download: Download full-size image

Table 2. Helicopter state codes and definitions.

Fig. 1. Different states of helicopters on patrol.
| State code | Definition |
| :--- | :--- |
| 0 | normal inspection flight process |
| 1 | inspection process to the alarm point |
| 2 | inspection process at the alarm point |


As shown in Tables 2 and in State 0, the helicopter executes the task according to the route planned by the offline scheduling. Currently, the route planning mainly considers how to efficiently traverse each patrol point. However, in States 1 and 2, factors such as the location of the alarm point, the current position of the helicopter, and the remaining battery power need to be considered to replan the flight route to the alarm point and the subsequent route. The transition among these three states and the corresponding decision-making logic constitutes the core dynamic scheduling mechanism of the model. After the helicopter explores the alarm point, if there is no second alarm point at this time, the command center will reschedule the flight path based on the existing point that has not been visited. Therefore, the process of the helicopter moving from the alarm point to the next inspection point is State 0. Note that only helicopters in State 0 can be dispatched to the emergencies.

Based on the discussion on the status of the helicopters, we build a two-stage dynamic path planning of the helicopter model. It is composed of two stages. Each stage has an independent objective function.
2.3. Stage 1 offline dispatch

$$
\begin{equation*}
f_{1}(x)=\sum_{i, j \in A_{k}} \sum_{k \in U} x_{i j}^{k} \cdot d_{i j} \tag{1}
\end{equation*}
$$

s.t. $\bigcup_{k=1}^{n} A_{k}=A$

$$
\begin{equation*}
\sum_{i \in A, i \neq j} \sum_{k \in U} x_{i j}^{k}=1 \tag{2}
\end{equation*}
$$

$$
\begin{equation*}
\sum_{j \in A, j \neq i} \sum_{k \in U} x_{i j}^{k}=1 \tag{3}
\end{equation*}
$$

$$
\begin{equation*}
\sum_{i \in A} x_{i i}=0 \tag{5}
\end{equation*}
$$

$$
\begin{equation*}
x_{i j}^{k} \cdot d_{i j} \leq h \tag{6}
\end{equation*}
$$

$$
\begin{equation*}
x_{i j}^{k} \in\{0,1\}, \forall i, j \in A, k \in U \tag{7}
\end{equation*}
$$

Eq. (1) is the objective function in the stage 1. At this stage, it is necessary to dispatch all the flight routes of the helicopters. Eq. (2) - Eq. (7) are the constraints. $A_{k}$ is the inspection points set of each helicopter $k$, where $k \in U$. Eq. (2) denotes that the number of total inspection points equals the sum of inspection points within each cluster. Eq. (3) and Eq. (4) represent that the path between two inspection points can only be flown once and cannot be reversed. Eq. (5) represents the next step from the helicopter inspection point must be other points. Due to the limited battery power of the helicopter, Eq. (6) limits the maximum flight distance of the helicopter. Eq. (7) indicates that the decision variable is a binary variable.

### 2.4. Stage 2 online dispatch with emergencies

In this stage, routes are known for the route scheduling routes of Stage 1. As a result, some of the parameters in Stage 2 are derived from Stage 1.

$$
\begin{equation*}
f_{2}(x)=\sum_{t \in T}\left(\sum_{k \in U-k_{\text {min }}} \sum_{i, j \in A_{k}} x_{i j}^{k} \cdot d_{i j}+\sum_{\beta \in B} d_{k_{\text {min }}, \beta}+\sum_{i, j \in M-S_{t}} x_{i j}^{k_{\text {min }}} \cdot d_{i j}\right) \tag{8}
\end{equation*}
$$

$$
\begin{equation*}
d_{k_{\text {min }}, \beta}=\min _{k \in U} d_{P_{k}, P_{\beta}}, k \in C\{0\} \tag{9}
\end{equation*}
$$

$$
\begin{equation*}
P_{k}=v t-\operatorname{sum}\left(\sum_{i, j \in A_{k}}^{Q} l_{i j}\right) \tag{10}
\end{equation*}
$$

$Q=Q_{t}$, if $\left.\bmod \left(v t, \sum_{Q_{t} \in W} \sum_{i j \in S_{t}} l_{i j}\right)=1\right)$
s. t. $t_{i} \leq \max (T), i=1,2, \ldots N$

$$
\begin{equation*}
k_{\text {min }} \leq n \tag{12}
\end{equation*}
$$

$$
\begin{equation*}
\sum_{Q_{t} \in W} \sum_{i j \in S_{t}} l_{i j} \leq v t \tag{13}
\end{equation*}
$$

$$
\begin{equation*}
\sum_{i \in A, i \neq j} \sum_{k \in U} x_{i j}^{k}=1 \tag{14}
\end{equation*}
$$

$$
\begin{equation*}
\sum_{i, j \in M-S} x_{i j}^{k_{\min }}=1 \tag{15}
\end{equation*}
$$

$$
\begin{equation*}
x_{i j}^{k} \cdot d_{i j} \leq h \tag{16}
\end{equation*}
$$

$$
\begin{equation*}
x_{i j}^{k} \in\{0,1\}, \forall i, j \in A, k \in U \tag{17}
\end{equation*}
$$

$x_{i, j}^{k_{\text {min }}} \in \operatorname{TSP}\left(M-S_{t}\right.$, start $\left.=\beta\right)$

Eq. (8)-Eq. (11) are the objective function in Stage 2. Eq. (8) is the overall objective function. It consists of three parts: the path that the helicopter has already traveled, the path from its current location to the alarm point, and the subsequent flight paths to the remaining points that have not yet been patrolled. Eq. (9)-Eq. (11) exhibits a specific calculation of the length of the helicopter's route from the current point to the alarm point. Eq. (12)-Eq. (18) are constraints. Eq. (12) requires that the time from helicopter takeoff for an alarm to occur must be less than the maximum time in the set $T$. Eq. (13) indicates that the number of helicopters dispatched must be less than the total number of helicopters. Eq. (14) represents that the total distances of the inspection points following the offline dispatch route up to the time of the alarm should be less than or equal to the total distances actually flown by the helicopters. Eq. (15) and Eq. (16) indicate that all helicopters can pass over the same route one time and only once. Eq. (17) limits the maximum flight distance of the helicopter. Eq. (18) indicates that the decision variable is a binary variable. Eq. (19) highlights that the TSP of remaining un-inspected points starts from emergency point $\beta$.

As shown in Fig. 2, the online scheduling phase of this model employs a dynamic route reconfiguration strategy, whose core logic is as follows: After the helicopter completes the inspection of emergency point $\beta, \beta$ is taken as the new starting point to re-solve the Traveling Salesman Problem (TSP) for the remaining un-inspected point set ( $M-S_{t}$ ). Ref (Yin et al., 2018). points out that traditional two-stage scheduling methods lead to efficiency losses due to "returning to original routes" involving long-distance backtracking. Stage 2 significantly improves emergency response efficiency by dynamically generating the shortest path in real-time through route reconfiguration.
![](https://cdn.mathpix.com/cropped/286e4446-4f36-48bc-9a51-5f151b677c98-07.jpg?height=660&width=1298&top_left_y=468&top_left_x=155)

Download: Download high-res image $(465 \mathrm{~KB})$
Download: Download full-size image
Fig. 2. Visualization of the entire model construction process.

This section implements the construction of a mathematical model for two-stage dynamic path planning considering emergencies. The time and location of the contingency are completely unknown. Therefore, temporary movements are only possible based on information about the occurrence of alarm points. Both Stage 1 and Stage 2 need to use optimization to obtain the optimal route when performing route planning. This will be shown in Section 3.

## 3. Methods

In order to solve the problem of anomaly information detection for forest fire rescue, based on Eq. (1) and Eq. (8), we propose a two-stage detection system for forest fire rescue in Fig. 3. The system is divided into three parts: online dispatch system, offline dispatch system, and the IACO algorithm solving the problem. The innovations mentioned in the paper we denote with highlighted "★" in Fig. 3. Both the online and offline dispatch use the $\underline{\mathrm{IACO}}$ algorithm to ensure their routes.

![](https://cdn.mathpix.com/cropped/286e4446-4f36-48bc-9a51-5f151b677c98-08.jpg?height=905&width=1449&top_left_y=93&top_left_x=150)
Fig. 3. Flow chart of the IACO algorithm in the two-stage dynamic route planning model.

Download: Download high-res image $(950 \mathrm{~KB})$
Download: Download full-size image

### 3.1. Cluster methods

In this section, all patrol points need to be classified according to the number of helicopters. For example, if we have three helicopters, we need to divide all the patrol points into three clusters. Each cluster should be separated as much as possible to ensure that the helicopter does not cross paths during actual flight and avoid collision risk. There are several common clustering methods.
(1) K-means

The K-means algorithm is a clustering algorithm proposed by J.B. MacQueen (Krishna and Murty, 1999). Its purpose is to divide a set of data into $k$ different clusters, so that the similarity of data in the same cluster is high, and the similarity between different clusters is as low as possible (Kodinariya and Makwana, 2013). Firstly, randomly select $k$ centroids, then assign each data point to the nearest centroid, and then update the centroid position to the mean of all points within their respective clusters until the centroid position no longer changes (Jie et al., 2020; Liu et al., 2023b). Fig. 4 shows the mechanism of K-means method.

![](https://cdn.mathpix.com/cropped/286e4446-4f36-48bc-9a51-5f151b677c98-09.jpg?height=982&width=1441&top_left_y=96&top_left_x=153)
Fig. 4. Mechanism of K-means method.

Download: Download high-res image $(674 \mathrm{~KB})$
Download: Download full-size image

The K-means algorithm is simple and easy to understand, computationally efficient, and suitable for processing large data sets. However, it needs to specify the number of clusters $k$ in advance. It is sensitive to the initial centroid selection, which affects the distribution of clusters and leads to local optimality.

## (2) Hierarchical Clustering

Hierarchical clustering represents hierarchical relationships of data by constructing a tree-like structure (Ran et al., 2023). There are two main approaches: agglomerative and divisive (Johnson, 1967). The agglomerative type starts with each data point and gradually merges the most similar points into a cluster until all data points are grouped into one cluster. The divisive type starts with all the points forming a cluster and gradually splits the cluster into smaller clusters (Bouguettaya et al., 2015). Fig. 5 shows the mechanism of Hierarchical clustering method.

![](https://cdn.mathpix.com/cropped/286e4446-4f36-48bc-9a51-5f151b677c98-09.jpg?height=619&width=1300&top_left_y=1927&top_left_x=155)
Fig. 5. Mechanism of Hierarchical clustering method.

Download: Download high-res image $(390 \mathrm{~KB})$
Download: Download full-size image

The hierarchical clustering method does not need to specify the number of clusters in advance and can provide the hierarchical structure of data, which is suitable for small data sets (Day and Edelsbrunner, 1984). However, its computational complexity is high and it is sensitive to noise and outliers. Once merged or split, it cannot be undone.

## (3) Density-Based Spatial Clustering of Applications with Noise (DBSCAN)

DBSCAN works by looking for dense areas of data points. It defines the dense region by two parameters: the minimum number of points per cluster MinPts and the neighborhood radius $\varepsilon$. DBSCAN divides points into core points (containing at least MinPts in the $\varepsilon$ neighborhood), boundary points, and noise points (Bushra and Yi, 2021). The core point forms a cluster with the points in its neighborhood, and the boundary points are connected to the core point to form a cluster (Qian et al., 2024). Fig. 6 shows the mechanism of DBSCAN clustering method.

![](https://cdn.mathpix.com/cropped/286e4446-4f36-48bc-9a51-5f151b677c98-10.jpg?height=708&width=1439&top_left_y=731&top_left_x=155)
Fig. 6. Mechanism of DBSCAN clustering method.

Download: Download high-res image $(543 \mathrm{~KB})$
Download: Download full-size image

DBSCAN can find clusters of any shape. It is suitable for processing noisy data, and can automatically identify the number of clusters (Kumar and Reddy, 2016). However, the choice of parameters ( $\varepsilon$ and MinPts) has a great influence on the results, resulting that it is not robust enough for high-dimensional data. It has high computational complexity and long computational time.

### 3.2. Selection of cluster methods

The emergency response to forest fires requires the rapid generation of a scheduling plan. The time complexity of the K-means algorithm is $O(n k t)$. (where $n$ is the number of samples, $k$ is the number of clusters, and $t$ is the number of iterations). When dealing with 100 inspection points, the time taken for a single clustering is only onefifth of that of hierarchical clustering.

In this study, the inspection problem is transformed into a multiple TSP question, and the partitioning results of Kmeans directly correspond to the inspection areas of helicopters. Compared with the DBSCAN clustering method, the clusters generated by K-means are more suitable for solving the TSP in terms of spatial distribution. Table 3 compares the different-dimensions of the cluster methods.

Table 3. Comparisons of the different-dimensions of the cluster methods.
| Index | K-means | Hierarchical Clustering | DBSCAN |
| :--- | :--- | :--- | :--- |
| Time Complexity | $O(n k t)$ | $O\left(n^{3}\right)$ | $O\left(n^{2}\right)$ |


| Index | K-means | Hierarchical Clustering | DBSCAN |
| :--- | :--- | :--- | :--- |
| Clustering Time for 100 Points (s) | 0.042 | 2.13 | 0.89 |
| Number of Path Intersections | 7 | 15 | 9 |
| Scalability (for 1000 Points) | Linearly Scalable | Memory Overflow | Exponential Growth |

Considering the emergencies in the forest area, we need to divide randomly generated inspection points based on the number of helicopters dispatched as soon as possible. Moreover, the number of inspection points is of medium size. Therefore, the K-means clustering method is very suitable for classification.

The dependence on the selection of initial values in the K-means method has always been a pain point. We have adopted various strategies to reduce its impact. In practical operation, we combine some heuristic methods to select the initial centroids. First, we calculate the central position of all inspection points. Then, based on this center, we select points that are relatively far away as the initial centroids according to certain rules, thereby improving the stability and accuracy of clustering. This approach can, to a certain extent, avoid the problem of poor clustering results caused by inappropriate selection of initial values. Algorithm 1 shows the improvement of the original K-means.

## Algorithm 1

Improvement of the K-means.

Input: Inspection Point Set $P$, Number of Helicopters $k$
1 Traverse all inspection points and calculate the sum of x -coordinates and the sum of y -coordinates:

$$
\begin{equation*}
x_{\text {total }}=\sum_{i=1}^{m} x_{i} \tag{20}
\end{equation*}
$$

$$
\begin{equation*}
y_{\text {total }}=\sum_{i=1}^{m} y_{i} \tag{21}
\end{equation*}
$$

2 Calculate the Geometric Center $O$ :
$O(x)=\frac{x_{\text {total }}}{m}(22)$
$O(y)=\frac{y_{\text {total }}}{m}(23)$
3 Initialize the Distance Array distance (m)
4 Traverse each inspection point $p_{j}$ :
distance $(m)=\sqrt{\left(p_{j}(x)-O(x)\right)^{2}+\left(p_{j}(y)-O(y)\right)^{2}}(24)$
5 Sort the inspection points in descending order of distance to generate the index sequence sorted _index
6 Select the points corresponding to the first $k$ indices as the initial centroids:

$$
\begin{equation*}
C= \tag{25}
\end{equation*}
$$

$[P($ sorted_index (1)), $P($ sorted_index (2) $), \ldots, P($ sorted_index $(k-1))]$
Output: Initial Centroid Set C

In the improved K-means initial centroid selection strategy of this study, through two key steps, namely the calculation of the geometric center and the selection based on distance priority, the clustering effect and computational efficiency have been effectively improved.

On the one hand, by calculating the geometric center of all the detection points as the origin, it is ensured that the initial centroids are distributed in the central area of the dataset. On the other hand, according to the distance priority criterion, the points are sorted in descending order of their distances from the origin, and the $k$ points with the farthest distances are selected as the initial centroids. Compared with the traditional random method, this approach can significantly enhance the stability of clustering.

### 3.3. Ant colony algorithm

(1) Path construction

Ant Colony Optimization algorithm (ACO) is an optimization algorithm proposed by Dorigo (Dorigo et al., 2006). It finds the optimal fitness by studying the intelligent behavior of ants foraging. The ACO algorithm has strong robustness, self-organization and distributed computing, etc. (Liao et al., 2014). Therefore, it is widely used in flight path planning. In the path planning problem, ACO is mainly composed of initialization pheromone, optimal solution construction and pheromone update (Huang and Lin, 2010; Wang et al., 2016).

Similar to GA, ACO takes a roulette wheel approach when selecting the next city to visit. $P_{i j}^{k}(t)$ represents the probability that ant $k$ moves from position $i$ to position $j$ at time $t$ :

$$
P_{i j}^{k}(t)=\left\{\begin{array}{c}\frac{\left[\tau_{i j}(t)\right]^{\alpha} *\left[\eta_{i j}(t)\right]^{\beta}}{\sum_{s \in \text { allowed }_{k}}\left[\tau_{i s}(t)\right]^{\alpha} *\left[\eta_{i s}(t)\right]^{\beta}}, \text { if } j \in \text { allowed }_{k}  \tag{26}\\ 0, \text { otherwise }\end{array}\right.
$$

Where allowed $_{k}=\left\{C-t a b u_{k}\right\}$ represents the set of cities that ant $k$ is allowed to choose in the next step. $t a b u_{k}$ is the tabu search table of ant $k$, which records the cities that ant $k$ has already visited. $\eta_{i j}$ is heuristic information, which represents the expected degree of transfer from city $i$ to city $j$. $\alpha$ and $\beta$ are weight factors, which reflect the amount of information accumulated by ants in the process of movement and the importance of heuristic information in ants' path selection.
(2) Path update

In the process of transmission, the pheromone on the path will gradually increase as individual ants continue to choose this path. But over time, some of the pheromones evaporate. Therefore, pheromone concentration is in a dynamic process. The amount of pheromone change can be recorded as $\Delta \tau_{i j}(t)$, and when the ant has traversed all the cities, the pheromone update is defined as follows (Liu et al., 2023a):

$$
\begin{equation*}
\tau_{i j}(t+1)=(1-\rho) * \tau_{i j}(t)+\Delta \tau_{i j}(t) \tag{27}
\end{equation*}
$$

s.t. $\left\{\begin{array}{c}\Delta \tau_{i j}(t)=\sum_{k=1}^{m} \Delta \tau_{i j}^{k}(t) \\ 0<\rho<1\end{array}\right.$

Where $\rho$ is the pheromone evaporation coefficient, which represents the attenuation degree of pheromone; ( $1-\rho$ ) is a pheromone residue factor, representing the degree of pheromone residue. $\Delta \tau_{i j}^{k}(t)$ represents the amount of pheromone ant $k$ remains on the path between city $i$ and city $j$ during this optimization process. The formula of $\Delta \tau_{i j}^{k}(t)$ is denoted in Eq. (28).

$$
\Delta \tau_{i j}^{k}(t)=\left\{\begin{array}{c}\frac{Q}{L_{k}}, \text { if ant } k \text { passes through path } i j  \tag{28}\\ 0, \text { otherwise }\end{array}\right.
$$

Where $Q$ represents the total amount of pheromone released by the ants in one cruise. $L_{k}$ is the total length of the route passed by the ant $k$ in this iteration. The pheromone increments method uses the global information and updates the pheromones on all paths after the ant completes one iteration, so it has better practical effect than other pheromone increments methods.

### 3.4. Improved ant colony algorithm

(1) Pheromone Initialization

In the traditional ant colony algorithm, the initial pheromone is uniformly distributed, which causes the ants to choose more blindly in the initial search process. The study of dynamically distributing the initial pheromone
content according to the Euclidean distance from the node to the line can make the initial search efficiency effectively improved. Choose $y=x^{-0.5}$ as the decay function of the initial pheromone, and the initial pheromone $\tau_{0}$ is denoted as:

$$
\begin{equation*}
\tau_{0}=\left(1+d_{i j k}^{-0.5}\right) n \tag{29}
\end{equation*}
$$

where $n$ denotes the number of ants, $d_{i j k}$ denotes the Euclidean distance of a node from the line connecting the initial and target points, and $\tau_{0}$ denotes a fixed pheromone constant value.
(2) Pheromone update methodology

Realistically, it is observed that certain aspects of ants' behavior in real life do not fully align with the algorithm's model. In natural ant colonies, ants exhibit a high degree of path diversity and are not restricted in their choices. However, in the context of the TSP, the situation differs. According to Eq. (26), the set allowed $_{k}=\{1,2, \ldots, n\}-t a b u_{k}$, gradually increases the elements in $t a b u_{k}$ and decreases the elements in allowed $_{k}$ as the cities visited by the ants increase until the end of the tour (Chen et al., 2024). As a result, the ants' path choices become increasingly constrained by the tabu list, leading to poorer path selection performance towards the end of the tour. Additionally, in ACO, pheromone updates are applied to all paths, including those that are suboptimal, which can interfere with the optimization process of subsequent ants (Wang and Han, 2021).

To address this issue, we propose a new pheromone updating strategy. This approach aims to reduce the influence of pheromones from the latter part of the tour on the search behavior of following ants. The key aspect of this new strategy is to enhance the pheromone update intensity during the early stages of the tour while diminishing the update intensity in the latter stages. The update formulation is shown in Eq. (30).

$$
\Delta \tau_{i j}^{k}=\left\{\begin{array}{c}Q \times \frac{n-s+1}{n(n+1) / 2}, \text { if ant } k \text { passes through path } i j  \tag{30}\\ 0, \text { otherwise }\end{array}\right.
$$

Where $n$ indicates the total number of ants and $s$ refers the current position of the ant in the queue. Eq. (30) describes the distribution of pheromones onto each path in linearly decreasing proportion according to the ants' traversal order. From an algorithmic design perspective, Eq. (30) already realizes the concept of pheromone decrement. However, when applied to real-world problems, the fixed proportion of pheromone distribution in Eq. (28) makes it difficult to achieve optimal performance. To address this limitation, we introduce a maximum pheromone update constant $\operatorname{Max}(\mathrm{C})$ and a minimum update constant $\operatorname{Min}(\mathrm{C})$, where $\operatorname{Max}(\mathrm{C})$, $\operatorname{Min}(\mathrm{C}) \in \mathbb{R}^{+}$.

$$
\Delta \tau_{i j}^{k}=\left\{\begin{array}{c}Q \times \frac{\operatorname{Max}(C)-\frac{(n-s+1)(\operatorname{Max}(C)-\operatorname{Min}(C))}{n}}{\frac{(n-1)(\operatorname{Max}(C)+\operatorname{Min}(C))}{2}}, \text { if ant } k \text { passes through path } i j  \tag{31}\\ 0, \text { otherwise }\end{array}\right.
$$

The purpose of setting these two constants is to ensure that pheromone update intensity decreases linearly with the progression of the search. Specifically, $\operatorname{Max}(\mathrm{C})$ is initialized to the maximum possible pheromone contribution of the best ant path, while $\operatorname{Min}(\mathrm{C})$ is set to a small positive value to maintain exploration. The determination process of specific parameters can be found in Section 3.5.4.
(3) Pseudocode

In this section, the pseudo-code of the proposed IACO algorithm solving the model is given. Since the whole model consists of two stages, the pseudo-code also consists of stage 1 and stage 2 . Stage 1 is the offline scheduling, which is mainly responsible for the route planning before the helicopter departs. Stage 2 is online scheduling, which is mainly responsible for the helicopter dispatching task for emergencies during inspections.
Algorithm 2

Pseudo-code of the IACO algorithm solving the mathematic model.

Stage 1: offline dispatch
Input: ants' number $n$, Max generation $G$, helicopter number $K, 100$ coordinates stored in the dataset, parameters of ACO $\alpha, \beta, \rho$, Pheromone update parameters $\operatorname{Max}(C), \operatorname{Min}(C)$

1 Determine the inspection points for each helicopter based on the number of helicopters $K$ by K-means clustering method.
Record the inspection points coordinates set $\operatorname{com}\{k\}$.
2 For $k=1: K$
3 Computed distance matrix according to the $\operatorname{com}\{k\}$
4 Initialize the pheromone matrix $\tau_{0}$ by Eq. (29)
5 For $i=1: G$
6 For $j=1: n$
7 Path selection based on pheromone concentration
8 Leave pheromone on passing paths for updating.
9 End for
10 Pheromone update for all paths
11 End for
12 Output the best path for the current cluster
13 End for
14 Output the optimal paths for all clusters and plot the offline scheduling map.
Stage 2: online dispatch
Input: the helicopter flight speed $v$, time of the alarm point $t$, coordinate of the alarm point $P$
1 Record the current status of all helicopters as state 0 until an alarm occurs.
2 Obtain the position of each helicopter based on $t, v$, and their current flight paths.
3 Select the helicopter closest to the alarm point and proceed immediately to inspect, and record the helicopter as state 1.
4 Helicopters in the process of inspection are recorded as state 2
5 Upon completing the inspection, the current helicopter position is recorded as the initial point of the TSP problem.
6 Use the IACO algorithm in Stage 1 to replan the flight path from the alarm point $\beta$ to remaining points $M-S_{t}$. Restore the current helicopter to state 0 .

7 If another alarm occurs
8 Judge the current status of the closest helicopter.
9 If current state $==1$ or current state $==2$
10 Dispatch of the second closest helicopter for inspection
11 Else
12 Repeat steps 1-6
13 End if
14 Output the final flight path for all the helicopters

### 3.5. Comprehensive test and analysis of IACO algorithm performance

### 3.5.1. Complexity analysis

## (1) Algorithm Complexity Analysis

The time complexity of traditional Ant Colony Optimization (ACO) algorithm is $O\left(m \cdot n^{2} \cdot G\right)$, which is determined by the number of ants $m$, patrol points $n$, and iterations $G$. IACO, building on ACO, uses dynamic initial pheromone allocation and phased pheromone update strategies.

In the offline scheduling phase, k - means clustering divides checkpoints into $k$ subsets (equal to the number of helicopters), each of size $n / k$. This reduces the single sub-problem complexity to $O\left(m \cdot(n / k)^{2} \cdot G\right)$ and the total to $O\left(m \cdot n^{2} \cdot n / k\right)$, a significant improvement. Although dynamic initial pheromone allocation requires an extra $O(n)$ Euclidean distance calculation, the orders of magnitude too small to be negligible.

In the online scheduling phase, when an emergency occurs, the nearest helicopter is selected and the path replanned. Assuming the number of emergency events is $N$ and the number of points for re-planning the path each time an emergency occurs are $M$, the complexity of a single emergency scheduling is $O\left(N \cdot M^{2} \cdot G\right)$. Since $M \ll n$, the overall complexity is controllable.

## (2). Resource Requirements in Practical Applications

The IACO algorithm enables real - time scheduling on ordinary computing devices like embedded systems or mobile terminals. For example, with 100 checkpoints and 3 helicopters, the offline scheduling phase takes around 5 s in our simulation experiments, and the online scheduling response time for local path adjustments is less than 1 s . Pre-calculation and caching mechanisms further reduce computational delays in emergency scenarios.

### 3.5.2. Benchmark test of IACO

To verify the effectiveness of the improvement we proposed, we conducted benchmark experiments on the Traveling Salesman Problem (TSP) using the Ant Colony Optimization (ACO) algorithm. In the experiments, we selected four cases from the classic benchmark library (Elloumi et al., 2014), namely Ulysses16, Eil51, Ch150, and Att532. The number of effective demand points in these cases increases successively from the smallest to the largest, which allows us to evaluate the solution effectiveness of the Improved Ant Colony Optimization (IACO) algorithm when dealing with TSPs of different dimensions.

For the sake of fairness, we carried out comparative experiments on the benchmark problems with seven other algorithms: traditional ACO, NNACS(Lalbakhsh et al., 2013), DACO(Lei and Shaoqiang, 2016), MACO(Lei and Shaoqiang, 2016), ACOU(Zhang and Zhang, 2017), MSPSO(Yang and Li, 2010), and IGA(Ab Wahab et al., 2024). Each experiment was run 30 times, and the average value of the experimental results was filled in. The population size was set to 30 . The experimental results are shown in Table 4.

Table 4. Results of different algorithms in TSP benchmark.
| Case | Distance of different algorithms in TSP benchmark (km) |  |  |  |  |  |  |  |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|  | IGA | MSPSO | ACO | NNACS | DACO | MACO | ACOU | IACO |
| Ulysses16 | 76.50 | 84.79 | 74.87 | 74.11 | 74.63 | 74.00 | 74.61 | 74.00 |
| Eil51 | 529.72 | 569.25 | 503.63 | 484.94 | 479.02 | 454.52 | 460.34 | 450.67 |
| Ch150 | 16454.79 | 17667.16 | 7393.77 | 6986.91 | 7724.39 | 7212.24 | 7012.09 | 6840.3 |
| Att532 | 539137.81 | 583515.61 | 132242.11 | 119771.09 | 142184.51 | 131914.27 | 127976.26 | 108906.82 |


In TSP benchmark experiments, the proposed IACO algorithm demonstrates excellent performance in both exploration and exploitation. Across cases like Ulysses16, Eil51, Ch150, and Att532, it efficiently explores the solution space, identifying superior solutions compared to IGA and MSPSO. In Ulysses16, its optimal result proves strong exploration of diverse solution areas. For exploitation, IACO effectively mines promising regions. It achieves the same optimal 74.00 km as MACO in Ulysses16, and in Att532, its 108,906.82km outperforms others, highlighting solution-refining strength. Overall, IACO balances exploration and exploitation: it discovers high-quality solutions across TSP scenarios while thoroughly exploiting promising regions, surpassing other algorithms in both navigating the solution space and deepening solution refinement.

Fig. 7 presents the performance of each algorithm on TSP benchmark test. Considering both the aesthetics of the image rendering and the complexity of the problem, we selected the Ch 150 case. This problem involves a relatively large number of points to be scheduled while maintaining a visually appealing presentation in the interface.

![](https://cdn.mathpix.com/cropped/286e4446-4f36-48bc-9a51-5f151b677c98-16.jpg?height=1226&width=1311&top_left_y=718&top_left_x=148)
Fig. 7. Routes of different algorithms in Ch150.

Download: Download high-res image (1MB)
Download: Download full-size image

Through visual analysis of TSP path planning diagrams, IACO's performance advantages emerge distinctly. When contrasted with IGA and MSPSO, IACO's path exhibits greater streamlining. The paths of IGA and MSPSO (Fig. 7 (1) and (2)) are marked by excessive detours and chaotic connections, reflecting inefficient solution-space exploration. In contrast, IACO leverages the ant colony algorithm's pheromone-driven cooperation mechanism-an inherent strength of ant colony algorithms in handling TSP-to iteratively optimize path selection, avoiding redundant searches.

When compared with other variant ACO algorithms (such as those in Fig. 7 (3), (5)-(7)), IACO's path (Fig. 7 (8)) demonstrates higher compactness and fewer erratic detours. While traditional or variant ACO algorithms retain suboptimal path segments, IACO's improved strategies-including enhanced pheromone update rules or adaptive parameter adjustments-effectively strengthen both exploration and exploitation. This balance between global search and local optimization enables IACO to generate more efficient paths. Visually, these differences highlight
that IACO's improvement strategies significantly enhance the basic ant colony algorithm framework, making it more proficient in addressing complex TSP problems.

### 3.5.3. Sensitive test of IACO

Algorithm sensitivity analysis is crucial for evaluating algorithm robustness under parameter variations and environmental uncertainties, ensuring model reliability and generalization capacity in practical applications. In this section, the control variable method is adopted for sensitivity analysis. A comparative analysis is carried out for the two improvements of the algorithm proposed in Section 3.4. Table 5 shows whether the current algorithm has the corresponding improvement strategies.

Table 5. The corresponding improvement strategies of IACO in sensitive test.
|  | Pheromone Initialization | Pheromone update methodology |
| :--- | :--- | :--- |
| ACO | × | × |
| IACO-1 | ✓ | × |
| IACO-2 | × | ✓ |
| IACO | ✓ | ✓ |


Fig. 8 shows the iterations of sensitive test in four cases. The TSP benchmark test problem in Section 3.4.2 is used. The population size of all algorithms is set to 30 , and the number of iterations is set to 1000 .

![](https://cdn.mathpix.com/cropped/286e4446-4f36-48bc-9a51-5f151b677c98-17.jpg?height=1076&width=1444&top_left_y=1280&top_left_x=150)
Fig. 8. Sensitive analysis of 4 TSP benchmark tests.

Download: Download high-res image $(624 \mathrm{~KB})$
Download: Download full-size image

Fig. 8 presents that the dynamic pheromone initialization (in IACO - 1) stands out in the early stage of the algorithm. Compared with the ACO algorithm, it can reduce the fitness value more quickly. In the early stage of the Ulysses16 and Eil51 tests, there is a large decline, enabling it to converge rapidly to a better fitness value, which is
conducive to quickly finding a relatively optimal solution in the early stage. The segmented pheromone update (in IACO-2), on the other hand, exerts its effect in the middle and later stages of the algorithm. It can finely adjust the search direction and prevent the algorithm from falling into a local optimum. Especially in complex tests of larger scales such as Ch150 and Att532, it can still continuously optimize the fitness value in the later stage of iteration, playing a significant role in improving the quality of the solution.

In order to comprehensively demonstrate the stability of the algorithm, Fig. 9 shows the comparison of the maximum values, minimum values, and average values of several algorithms in tests of 30 runs.

![](https://cdn.mathpix.com/cropped/286e4446-4f36-48bc-9a51-5f151b677c98-18.jpg?height=886&width=1449&top_left_y=500&top_left_x=148)
Fig. 9. Comparison of update strategies in TSP benchmark.

Download: Download high-res image $(395 \mathrm{~KB})$
Download: Download full-size image

As shown in Fig. 9, compared with ACO, both IACO-1 and IACO-2 have improved performance in terms of the mean value, minimum value, and maximum value in each test, indicating that both improvement strategies are effective. In comparison with ACO, both IACO - 1 and IACO - 2 have optimized these indicators to varying degrees, which shows that both improvement strategies are effective. IACO-1 reduces the result fluctuations in some tests, while IACO-2 makes the overall values better. The IACO algorithm performs the best in all tests in terms of the three indicators, with stable results and low values, fully demonstrating the effectiveness of combining the two strategies.

### 3.5.4. Parameter test

To evaluate the robustness of the Improved Ant Colony Optimization (IACO) algorithm to its key parameters Max(C) and $\operatorname{Min}(\mathrm{C})$, a parametric study was conducted. The experiment focused on the offline scheduling phase of the twostage model, using the total flight distance as the performance metric. The population size of tests is set to 30 , and the number of iterations is set to 1000.The parameters were tested over the following ranges:
$\operatorname{Max}(\mathrm{C}) \in\{1,2,3,4,5\}$
$\operatorname{Min}(C) \in\{0.1,0.3,0.5,0.7,0.9\}$
For each parameter combination, the algorithm was executed 30 times on the Ch150 TSP benchmark test. The mean and standard deviation (std) of the total flight distance were recorded (Table 6, Table 7).

Table 6. Performance of IACO with varying $\operatorname{Max}(\mathrm{C})(\operatorname{Min}(\mathrm{C})=0.5)$.

| Max(C) | Mean Distance (km) | Std (km) |
| :--- | :--- | :--- |
| 1 | 6923.4 | 127 |
| 2 | 6840.3 | 89 |
| 3 | 6925.8 | 151 |
| 4 | 6931.2 | 183 |
| 5 | 7021.5 | 202 |


Table 7. Performance of IACO with varying $\operatorname{Min}(\mathrm{C})(\operatorname{Max}(\mathrm{C})=2)$.
| Min(C) | Mean Distance (km) | Std (km) |
| :--- | :--- | :--- |
| 0.1 | 6918.7 | 114 |
| 0.3 | 6945.2 | 101 |
| 0.5 | 6840.3 | 89 |
| 0.7 | 6914.5 | 98 |
| 0.9 | 7117.6 | 123 |


The optimal performance was achieved at $\operatorname{Max}(\mathrm{C})=2$ and $\operatorname{Min}(\mathrm{C})=0.5$, yielding a mean distance of 6840.3 km . This combination balances exploration (via $\operatorname{Min}(\mathrm{C})$ ) and exploitation (via $\operatorname{Max}(\mathrm{C})$ ) effectively. While the results vary slightly across parameter values, the differences are marginal. For example, increasing $\operatorname{Max}(\mathrm{C})$ beyond 2 leads to a gradual increase in mean distance due to excessive exploitation of suboptimal paths. Conversely, Min(C) values below 0.5 reduce exploration, resulting in higher variability (e.g., $\operatorname{Min}(\mathrm{C})=0.1$ has a std of 114 km ). These findings indicate that the IACO algorithm is robust to parameter variations within the tested ranges, with $\operatorname{Max}(\mathrm{C})=2$ and $\operatorname{Min}(C)=0.5$ providing the best trade-off between solution quality and stability.

## 4. Experiments

### 4.1. Experiments analysis

Considering the complexity and multi-scene possibilities of the forest field, in this section, we set up two experiments to fully simulate all possibilities of helicopters scouting when emergencies happen in the forest field.

First, we designed a basic independent regional helicopter dispatch. An alarm point occurs in the area where the helicopter is responsible and the current helicopter needs to be dispatched to the disaster point to rescue it. After the inspection, the optimal patrol route needs to be redesigned according to the situation of the remaining points.

Second, we designed a cross-area dispatch in a combined multi-helicopters crossover dispatch. Suppose an alarm point occurs during the current helicopter's patrol. The helicopter that is responsible for patrolling this area cannot be dispatched to inspect it based on the current state. At this point, the command center is asked to request that the nearest helicopter be dispatched across the region to inspect the area. Compared to the first case, multiple alarm points can interfere with multiple circuit paths, giving greater difficulty to the dispatch system.

The first case emphasizes path transfer after visiting the alarm point, while the second demonstrates coordinated assignments within the dispatch system. Both cases reflect potential real-world forest inspection situations and enhance preparedness for subsequent emergency missions. The periphery of the simulated forest field is a $100 \mathrm{~km} * 100 \mathrm{~km}$ rectangle, and 100 points that need daily inspection are randomly generated in each case.

### 4.2. Assumptions

To ensure the validity and applicability of the proposed two-stage dynamic path planning model, the following assumptions are made.
(1) All helicopters are assumed to maintain consistent operational parameters (e.g., speed, fuel capacity, payload) throughout both offline and online scheduling phases, neglecting mechanical failures or fuel consumption dynamics.
(2) All helicopters are pre-deployed to inspection regions and operate at full capacity, with no standby resources assumed at dispatch centers.
(3) The geographical coordinates of inspection points and emergency locations are fixed during offline planning, and no environmental changes are considered during real-time operations.
(4) The command center has instantaneous access to accurate, real-time data on helicopter positions, states, and emergency occurrences, with no communication delays or signal disruptions.
(5) All helicopters dispatched in the offline phase are assumed to be fully utilized, with no reserve drones available for immediate deployment. This ensures that reallocation decisions strictly rely on existing operational units.
(6) Flight distances and durations between points are calculated based on Euclidean distances and constant speeds, ignoring air currents, altitude variations, or other aerodynamic factors.

### 4.3. Independent regional dispatch

In this section, 100 points are randomly generated as daily inspection points. These points are used in offline scheduling to obtain the helicopter's flight path. Time of alarm point is randomly generated that the command center has received an emergency which requires an immediate helicopter inspection. The coordinate of the alarm point is randomly generated within the forest area. The command center needs to quickly dispatch the helicopter currently closest to the alarm point to the scene after the emergencies occurred. In this case, the random time of the alarm was 1.09 h after the helicopter began its patrol. The coordinates of the alarm point are [87.71,84.06]. Helicopter flying at $60 \mathrm{~km} / \mathrm{h}$ constant speed. Table 8 shows the parameters of different moments explored by helicopters.

Table 8. Parameters of different moments explored by helicopters.
| Helicopter Serial No. | ACO |  | IACO |  |  |  |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|  | I | II | III | I | II | III |
| Position of the scouting point before the emergency | [62.15, 22.543] | [30.11, 21.58] | [45.60, 70.55] | [71.41,25.49] | [25.06,41.98] | [38.08,57.78] |
| Current position of the helicopter at the time of the emergency | [57.41, 22.67] | [31.13, 14.90] | [43.49, 66.96] | [70.41,23.21] | [23.31,44.28] | [34.41,60.00] |
| Traveled distance (km) | 65.42 | 65.42 | 65.42 | 65.42 | 65.42 | 65.42 |
| Distance from the alarm point (km) | 68.46 | 89.36 | 47.42 | 63.27 | 75.70 | 58.48 |
| Number of inspection points | 34 | 33 | 33+1 | 34 | 33 | 33+1 |


In our helicopter flight simulation, three helicopters are in operation by the command center. The 100 inspection points were divided into three clusters of 34,33 , and 33 by K-means clustering to serve as probe points for each of the three helicopters. The position of the scouting point before the emergency represents the position of the helicopter immediately preceding the previous spotting point when the emergency occurred. According to the
distance from the alarm point, both the ACO and IACO algorithms indicate that Helicopter III is currently the closest to the alarm point, so priority is given to dispatching Helicopter III to the alarm point.

Fig. 10 shows the distance traveled for the time spent by the helicopter in case of only 1 alarm point. IACO algorithm with better optimization search when facing the same classified triple cluster points. Comparing the clusters individually, it is still the IACO algorithm that performs better, with a shorter distance traveled for the planned routes. Specific details of the comparison are demonstrated in Fig. 11.

![](https://cdn.mathpix.com/cropped/286e4446-4f36-48bc-9a51-5f151b677c98-21.jpg?height=810&width=1291&top_left_y=464&top_left_x=157)
Fig. 10. Helicopter flight time and route distance with different algorithms in case 1.

Download: Download high-res image $(491 \mathrm{~KB})$
Download: Download full-size image
![](https://cdn.mathpix.com/cropped/286e4446-4f36-48bc-9a51-5f151b677c98-21.jpg?height=378&width=479&top_left_y=1564&top_left_x=242)
a(1) Routes of helicopters offline dispatch by ACO
![](https://cdn.mathpix.com/cropped/286e4446-4f36-48bc-9a51-5f151b677c98-21.jpg?height=430&width=530&top_left_y=2090&top_left_x=214)
b(1) Routes for helicopter III emergency exploration by ACO
![](https://cdn.mathpix.com/cropped/286e4446-4f36-48bc-9a51-5f151b677c98-21.jpg?height=385&width=483&top_left_y=1561&top_left_x=910)
a (2) Routes of helicopters offline dispatch by IACO

![](https://cdn.mathpix.com/cropped/286e4446-4f36-48bc-9a51-5f151b677c98-21.jpg?height=420&width=502&top_left_y=2096&top_left_x=898)
b(2) Routes for helicopter III emergency exploration by IACO

Fig. 11. Collection of helicopter dispatch diagrams.
In the TSP problem, there is an important metric to judge the merit of route planning, which is that the routes should appear to have fewer crossings. Apparently, comparing Fig. 11a(1) and Fig. 11a(2) reveals that there are fewer line crossings in 11a (2). Combining the total route lengths of the two algorithms in Fig. 10, it can be deduced that the IACO algorithm has higher convergence accuracy and better performance. Fig. 11b shows the full two-stage dispatch route for Helicopter III. First, the helicopter patrols according to the offline dispatch route. Later, the emergency occurs, and it is temporarily requested by the command center to visit the alarm point for inspection. After the inspection, it immediately counts the current points that have not been inspected and regenerate the scouting route. Finally, return to the command center.

### 4.4. Cross-region joint helicopter dispatch

In Case 4.4, we design a cross-region scheduling model for a multi-helicopter joint crossing scenario. Suppose that three-alarm points go off sequentially while the helicopter is on patrol. The alarm points can be reported relatively distantly from each other or closely together. The command center needs to determine which helicopter is dispatched to the alarm points based on the current state and location of the helicopters. Table 9 shows the location and time of the alarms. In this scenario, up to 3 emergency points were randomly triggered across different clusters, and the model dynamically allocated helicopters based on real-time availability and proximity. The response mechanism prioritizes the nearest idle helicopter to each new emergency, ensuring efficient resource distribution even under concurrent demands.

Table 9. The location and time of the alarms.
|  | Location | Time of alarm after departure |
| :--- | :--- | :--- |
| Alarm point 1 | [51.86,96.45] | 1.48 |
| Alarm point 2 | [32.82,26.92] | 2.13 |
| Alarm point 3 | [23.80,49.37] | 2.60 |


The following Table 10 denotes the parameters for scheduling using the ACO and IACO algorithms.

Table 10. Parameters of different moments explored by helicopters.
| Helicopter Serial No. |  | ACO | IACO |  |  |  |  |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|  |  | I | II | III | I | II | III |
| Alarm point 1 | Position of the scouting point before the emergency | [90.81,21.24] | [22.65,47.08] | [23.33,82.49] | [90.45,13.51] | [19.19,40.56] | [1.44,67.78] |
|  | Current position of the helicopter at the time of the emergency | [91.53,25.49] | [20.43,46.84] | [21.05,80.87] | [90.66,18.17] | [90.72,19.34] | [22.62,47.03] |
|  | Traveled distance (km) | 89.0706 | 89.0706 | 89.0706 | 89.0706 | 89.0706 | 89.0706 |
|  | Distance from the alarm point (km) | 87.6798 | 57.4227 | 43.7786 | 86.3458 | 57.4227 | 41.2175 |
| Alarm point 2 | Position of the scouting point before the emergency | [79.75,29.50] | [6.70,19.77] | [51.86,96.45] | [79.75,29.50] | [9.66,22.77] | [51.86,96.45] |
|  | Current position of the helicopter at the time of the emergency | [79.04,31.34] | [8.64,21.74] | [72.01,104.35] | [78.92,31.68] | [78.50,32.77] | [8.65,21.74] |


| Helicopter Serial No. |  | ACO |  |  | IACO |  |  |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|  |  | I | II | III | I | II | III |
|  | Traveled distance (km) | 127.7529 | 127.7529 | 127.7529 | 127.7529 | 127.7529 | 127.7529 |
|  | Distance from the alarm point (km) | 46.4304 | 24.7259 | 86.7816 | 46.0457 | 24.7259 | 111.7561 |
| Alarm point 3 | Position of the scouting point before the emergency | [97.33,59.34] | [32.82,26.92] | [60.61,80.73] | [97.33,59.34] | [9.66,22.77] | [51.86,96.45] |
|  | Current position of the helicopter at the time of the emergency | [89.81,68.69] | [40.17,22.52] | [56.46,86.72] | [88.85,69.89] | [23.74,27.49] | [55.70,97.96] |
|  | Traveled distance (km) | 156.2744 | 156.2744 | 156.2744 | 156.2744 | 156.2744 | 156.2744 |
|  | Distance from the alarm point (km) | 68.77 | 39.25 | 48.42 | 68.20 | 31.45 | 58.12 |
| Number of inspection points |  | 42 | 27+2 | 31+1 | 42 | 27+1 | 31+2 |

According to Table 10, the ACO algorithm optimizes the helicopter dispatch numbers of III, II, and II. The first alarm point is inspected by helicopter III. The second alarm point is visited by helicopter II. The third alarm point is also inspected by helicopter $\|$. We have to pay special attention to the status of helicopter $\|$ at the time the third alarm occurred. It has been examined that helicopter $\|$ had already completed its mission to alarm point 2 with state 0 when alarm point 3 began to sound and that helicopter II was closest to alarm point 3 at that time. Therefore, helicopter II was dispatched to alarm point 3.

In IACO, the helicopter's dispatch number is III, II and III. During the third anomaly detection, although Helicopter II was closest to alarm point 3at the current time, Helicopter II was on its way to alarm point 2at this time with state 1. With the intention of obtaining the alarm information as soon as possible, Helicopter III, which was the second closest to the fire point, was dispatched to inspect alarm point 3 . In order to more clearly clarify the connection between offline and online scheduling, Fig. 12 shows the actual and planned distances for helicopters.

![](https://cdn.mathpix.com/cropped/286e4446-4f36-48bc-9a51-5f151b677c98-23.jpg?height=446&width=708&top_left_y=1747&top_left_x=155)
(1) ACO algorithm

![](https://cdn.mathpix.com/cropped/286e4446-4f36-48bc-9a51-5f151b677c98-23.jpg?height=446&width=696&top_left_y=1747&top_left_x=891)
(2) IACO algorithm

Download: Download high-res image $(392 \mathrm{~KB})$
Download: Download full-size image

Fig. 12. Helicopter flight time and route distance with different algorithms in case 2.

Fig. 12 compares the performance of ACO and IACO algorithms in terms of helicopter flight time and route distance. In the ACO algorithm, significant discrepancies exist between actual helicopter flight distances and flight times versus offline scheduling estimates. For example, Helicopter III recorded an actual flight distance of 425.9 km and time of 7.10h, far exceeding estimated values. In contrast, the IACO algorithm substantially reduces these deviations: actual flight distances and times for each helicopter more closely align with estimates. Taking

Helicopter III as an example, actual distance decreased to 399 km and time to 6.50 h , demonstrating a notable reduction in discrepancy. Overall, the IACO algorithm achieves more accurate alignment between planned expectations and real-world execution in flight scheduling, narrowing the gap between offline estimates and actual performance. This highlights its advantages in optimizing flight distance/time parameters while enhancing both scheduling efficiency and accuracy.

Fig. 13 shows the offline dispatch routes for all helicopters. Although the locations of the three-alarm points are the same in both algorithms, the serial numbers of the helicopters that are eventually dispatched are different because the helicopters are dispatched on different routes in different algorithms. Comparing Fig. 13(1) and Fig. 13(2) it is easy to see that, like cases 1 and 2, the flight routes solved by the ACO algorithm always cross. This also means that the ACO algorithm fails to find a way to pass through neighboring points without overlapping paths. In Fig. 13(2), the current position of the helicopter closest to the alarm point 3 is not on the offline dispatch route because helicopter 3 had already been dispatched to alarm point 1 before. Therefore, it was not at the predicted position in the offline dispatch diagram at that time.

![](https://cdn.mathpix.com/cropped/286e4446-4f36-48bc-9a51-5f151b677c98-24.jpg?height=685&width=1306&top_left_y=832&top_left_x=153)
Fig. 13. The offline dispatch routes for all helicopters.

Fig. 14 shows the routes that helicopters are assigned to travel to view alarm points. Fig. 14a and b represent the respective routes of the helicopters conducting the mission, which need to be compared horizontally. Fig. 14a(2) and 14a (3) are both performed by helicopter II. Therefore, they have multiple similarities. It is observed that the previous inspection point of the alarm point in 14a (3) is the alarm point in 14a (2). In other words, after inspecting alarm point 2, helicopter II proceeds directly to alarm point 3 without passing any other inspection point. Additionally, compared to flying according to the original route, the path reconstruction method reduces redundant routes by $7.6 \%$, demonstrating the effectiveness of the proposed online scheduling path reconstruction strategy.

![](https://cdn.mathpix.com/cropped/286e4446-4f36-48bc-9a51-5f151b677c98-25.jpg?height=835&width=1453&top_left_y=96&top_left_x=148)
Fig. 14. Routes of helicopters patrol to the alarm points.

Download: Download high-res image $(690 \mathrm{~KB})$
Download: Download full-size image

Fig. $14 \mathrm{~b}(1)$ and $14 \mathrm{~b}(3)$ are both performed by helicopter III. The red line in $14 \mathrm{~b}(1)$ is the same as the one in $14 \mathrm{~b}(3)$. Comparatively, the blue line, which represents the route of the remaining points of Helicopter 2 in Fig. 14b(3) is significantly different from the one in Fig. 14b(1) after it finishes the inspection. It is because as the helicopter proceeds to the next inspection point, redispatch is performed quickly to ensure that the helicopter is able to return to the command center by the shortest possible path. In an overall comparison between Fig. 14a and $b$, the routes in Fig. 14a have more crossings, which indicates that the ACO algorithm does not find the optimal route. The IACO algorithm, on the other hand, finds routes with fewer curvatures in comparison.

### 4.5. Discussion

## (1) Limitations

The experimental design of this study focuses on two typical scenarios: independent regional scheduling and crossregional joint scheduling. However, certain limitations remain in practical applications.

First, the case assumes all helicopters operate at full operational capacity without considering collaborative scheduling with backup drones or ground-based rescue forces. Second, the experimental scenarios only address single emergency incidents, without simulating multiple simultaneous alerts or cascading effects (such as secondary disasters caused by fire spread). Additionally, the model assumes a static geographic environment, neglecting impacts on helicopter flights from weather variations and dynamic terrain changes.

Due to the extreme complexity of the forest environment, these cases cannot cover all possible situations. In actual forest fire rescue, complex scenarios may still occur, such as the simultaneous appearance of multiple fire alarm points that are extremely scattered, and significant differences in terrain and climate conditions among different regions, which have not been involved in this experiment.

In the scenario of multiple scattered fire alarm points, the competition and allocation of helicopter resources will be more complicated, which may lead to a decrease in the response speed and resource utilization efficiency of the scheduling strategy proposed in this study. For regions with significant differences in terrain and climate conditions, the flight performance and path planning of helicopters will face greater challenges, and the existing scheduling model based on distance and simple environmental assumptions may need further optimization.
(2) Extensibility

The two-stage model demonstrates significant potential for expanded applications. First, by integrating historical fire data, weather forecasts, or machine learning models to establish a spatiotemporal prediction framework, predictive scheduling can be achieved: offline phases prioritize coverage of high-risk areas, while online phases dynamically adjust response priorities using real-time fire data. Second, integration with existing forest fire management platforms (e.g., the European Forest Fire Information System (EFFIS)) enables multisystem collaboration: directly outputting route planning results to UAV control systems or triggering intelligent scheduling through IoT sensors. Additionally, the model can be extended to helicopter-UAV cooperative rescue networks, leveraging UAV agility for reconnaissance while helicopters handle material transportation. This requires addressing challenges related to heterogeneous device communication protocols and task allocation. These expansion directions not only enhance the model's practical application value but also provide theoretical support for constructing smart forest fire management systems. In future research, we will conduct further investigations into these aspects.

## 5. Conclusion

This study addresses the dynamic scheduling challenges in forest helicopter inspections through a two-stage path planning model and an Improved Ant Colony Optimization (IACO) algorithm, achieving three key advancements.
(1) Two-stage dynamic path planning model. A novel two-stage architecture integrating offline prescheduling and online dynamic adjustments overcomes traditional static planning limitations. By decomposing the global VRP into clustered TSPs via K-means clustering, computational complexity is significantly reduced. The real-time emergency response mechanism reconstructs optimal paths from alarm points to pending inspection nodes, experimentally demonstrating a $7.6 \%$ reduction in flight distance compared to conventional return-to-base strategies.
(2) Enhanced IACO Algorithm. The proposed IACO algorithm introduces a dynamic pheromone initialization strategy guided by Euclidean distances from nodes to target lines, coupled with a phased pheromone intensity decay mechanism. By balancing exploration and exploitation through $\operatorname{Max}(\mathrm{C}) / \operatorname{Min}(\mathrm{C})$ parameters, it effectively mitigates premature convergence issues inherent in traditional ACO methods.
(3) Multimodal Scenario Validation. Comprehensive validation across single-helicopter regional dispatch and multi-helicopter cross-cluster coordination scenarios confirms the framework's adaptability. The IACO algorithm achieves $4.1 \%$ shorter offline routes and $7.3 \%$ faster emergency responses, particularly excelling in multi-agent synchronization through emergency point relocation strategies.

This work establishes a methodological framework balancing routine efficiency and emergency resilience for forest inspection systems, providing foundational insights for intelligent fire rescue infrastructure. Although the twostage dynamic route planning model and IACO algorithm proposed in this study demonstrate certain advantages in simulated experimental scenarios, due to the limitations of the experimental settings, both the model and algorithm may require adjustment and optimization according to specific conditions when applied to different forest environments. Our future research will further expand and refine this model for broader forest fire rescue scenarios.

## CRediT authorship contribution statement

Xu Wangying: Writing - original draft, Software, Methodology, Data curation. Xie Naiming: Writing - review \& editing, Conceptualization.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Acknowledgment

This work was supported by National Natural Science Foundation of China under grant 72171116, T2441003 and 92367301, the Fundamental Research Funds for the Central Universities, China under grant NP2024203, "333 talent" project in Jiangsu Province (China).

Recommended articles

## Data availability

Data will be made available on request.

## References

Ab Wahab et al., 2024 M.N. Ab Wahab, A. Nazir, A. Khalil, W.J. Ho, M.F. Akbar, M.H.M. Noor, A.S.A.J.E.S.w.A. Mohamed
Improved Genetic Algorithm for Mobile Robot Path Planning in Static Environments, vol. 249 (2024), Article 123762
View PDF View article
View in Scopus r Google Scholar

Ajith and Jolly, 2021 V. Ajith, K. Jolly
Unmanned aerial systems in search and rescue applications with their path planning: a review
Journal of Physics: Conference Series, IOP Publishing (2021), Article 012020
Crossref π View in Scopus π Google Scholar π

Al-Hilo et al., 2020 A. Al-Hilo, M. Samir, C. Assi, S. Sharafeddine, D. Ebrahimi
UAV-assisted content delivery in intelligent transportation systems-joint trajectory planning and cache management
IEEE Trans. Intell. Transport. Syst. (2020), pp. 1-13
Google Scholar π

B et al., 2021 Y.W.A. B, K.H.L. C, X.H. D
Trajectory-based flight scheduling for AirMetro in urban environments by conflict resolution
Transport. Res. C Emerg. Technol., 131 (2021), Article 103355
Google Scholar π

Bektas, 2006 T.J.o. Bektas
The Multiple Traveling Salesman Problem: an Overview of Formulations and Solution Procedures, vol. 34 (2006), pp.
209-219
View PDF
View article
View in Scopus r
Google Scholar π

Bouguettaya et al., 2015 A. Bouguettaya, Q. Yu, X. Liu, X. Zhou, A. Song
Efficient agglomerative hierarchical clustering
Expert Syst. Appl., 42 (2015), pp. 2785-2797
View PDF View article View in Scopus r Google Scholar

Brown et al., 2020 T. Brown, S. Leach, B. Wachter, B. Gardunio
The Northern California 2018 extreme fire season
Bull. Am. Meteorol. Soc., 101 (2020), pp. S1-S4
Crossref $\lambda$ View in Scopus $\lambda$ Google Scholar $\lambda$

## Comparative analysis review of pioneering DBSCAN and successive density-based clustering algorithms

IEEE Access, 9 (2021), pp. 87918-87935
Crossref $\lambda$ View in Scopus $\lambda$ Google Scholar $\lambda$
Carvalho A et al., 2024 E. Carvalho A, J.F. Ferreira, D. Portugal
3D traversability analysis and path planning based on mechanical effort for UGVs in forest environments
Robot. Autonom. Syst., 171 (2024)
104560
https://doi.org/10.1016/j.robot.2023.104560 π
Google Scholar π
Chen et al., 2024 E. Chen, Z. Zhou, R. Li, Z. Chang, J. Shi
The multi-fleet delivery problem combined with trucks, tricycles, and drones for last-mile logistics efficiency requirements under multiple budget constraints
Transp. Res. Part E, 187 (2024)
Google Scholar π
Day and Edelsbrunner, 1984 W.H. Day, H. Edelsbrunner
Efficient algorithms for agglomerative hierarchical clustering methods
J. Classif., 1 (1984), pp. 7-24

View in Scopus r Google Scholar $\lambda$
Dorigo et al., 2006 M. Dorigo, M. Birattari, T. Stutzle
Ant colony optimization
IEEE Comput. Intell. Mag., 1 (2006), pp. 28-39
View in Scopus $\lambda$ Google Scholar $\lambda$
Ebrahimi et al., 2021 D. Ebrahimi, S. Sharafeddine, P.H. Ho, C. Assi
Autonomous UAV trajectory for localizing ground objects: a reinforcement learning approach
IEEE Trans. Mobile Comput., 20 (2021), pp. 1312-1324
Crossref $\lambda$ View in Scopus $\lambda$ Google Scholar $\lambda$
Elloumi et al., 2014 W. Elloumi, H. El Abed, A. Abraham, A.M.J.A.S.C. Alimi
A Comparative Study of the Improvement of Performance Using a PSO Modified by ACO Applied to TSP, vol. 25 (2014), pp. 234-241

View PDF
View article
View in Scopus π
Google Scholar π

Giannakidou et al., 2024 S. Giannakidou, P. Radoglou-Grammatikis, T. Lagkas, V. Argyriou, S. Goudos, E.K. Markakis, P.
Sarigiannidis
Leveraging the power of internet of things and artificial intelligence in forest fire prevention, detection, and restoration: a comprehensive survey
Internet of Things, 26 (2024), Article 101171
View PDF
View article
View in Scopus
Google Scholar

Guha et al., 2022 S. Guha, R.K. Jana, M.K. Sanyal
Artificial neural network approaches for disaster management: a literature review
Int. J. Disaster Risk Reduct., 81 (2022), Article 103276
View PDF
View article
View in Scopus
Google Scholar フ
Huang and Lin, 2010 S.H. Huang, P.C. Lin
A modified ant colony optimization algorithm for multi-item inventory routing problems with demand uncertainty
Transp. Res. Part E, 46 (2010), pp. 598-611

Jazebi et al., 2019 S. Jazebi, F. De Leon, A. Nelson
Review of wildfire management techniques-Part I: causes, prevention, detection, suppression, and data analytics
IEEE Trans. Power Deliv., 35 (2019), pp. 430-439
Google Scholar π

Jie et al., 2020 C. Jie, Z. Jiyue, W. Junhui, W. Yusheng, S. Huiping, L. Kaiyan
Review on the research of K-means clustering algorithm in big data
2020 IEEE 3rd International Conference on Electronics and Communication Engineering (ICECE), IEEE (2020), pp. 107-111
Crossref $\lambda$ View in Scopus $\lambda$ Google Scholar $\lambda$

Johnson, 1967 S.C. Johnson
Hierarchical clustering schemes
Psychometrika, 32 (1967), pp. 241-254
View in Scopus $\lambda$ Google Scholar $\lambda$

Kodinariya and Makwana, 2013 T.M. Kodinariya, P.R. Makwana
Review on determining number of cluster in K-means clustering
Int. J., 1 (2013), pp. 90-95
Google Scholar π

Krishna and Murty, 1999 K. Krishna, M.N. Murty
Genetic K-means algorithm
IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), 29 (1999), pp. 433-439
View in Scopus $\lambda$ Google Scholar π

Kumar and Reddy, 2016 K.M. Kumar, A.R.M. Reddy
A fast DBSCAN clustering algorithm by accelerating neighbor searching using Groups method
Pattern Recogn., 58 (2016), pp. 39-48
Google Scholar π

Kyrkou et al., 2022 C. Kyrkou, P. Kolios, T. Theocharides, M. Polycarpou
Machine learning for emergency management: a survey and future outlook
Proc. IEEE, 111 (2022), pp. 19-41
Google Scholar π

Lalbakhsh et al., 2013 P. Lalbakhsh, B. Zaeri, A. Lalbakhsh
An improved model of ant colony optimization using a novel pheromone update strategy
IEICE Transactions on Information and Systems, E96.D (2013), pp. 2309-2318
View in Scopus r Google Scholar r

Lei and Shaoqiang, 2016 L. Lei, W. Shaoqiang
An improved ant colony optimization algorithm using local pheromone and global pheromone updating rule
2016 International Conference on Intelligent Transportation, Big Data \& Smart City (ICITBS) (2016), pp. 63-67
View in Scopus $\lambda$ Google Scholar $\lambda$

Li and Cheng, 2023 Y. Li, H. Cheng
Unidirectional-road-Network-based global path planning for cleaning robots in semi-structured environments

2023 IEEE International Conference on Robotics and Automation (ICRA) (2023), pp. 1572-1578
Crossref л View in Scopus л Google Scholar л

## A unified ant colony optimization algorithm for continuous optimization

Eur. J. Oper. Res., 234 (2014), pp. 597-609
View PDF
View article
View in Scopus r
Google Scholar フ

Liu et al., 2023a C. Liu, L. Wu, W. Xiao, G. Li, D. Xu, J. Guo, W. Li
An improved heuristic mechanism ant colony optimization algorithm for solving path planning
Knowl. Base Syst., 271 (2023), Article 110540
View PDF
View article
View in Scopus
Google Scholar π

Liu et al., 2023b H. Liu, J. Chen, J. Dy, Y. Fu
Transforming complex problems into K-means solutions
IEEE Trans. Pattern Anal. Mach. Intell., 45 (2023), pp. 9149-9168
View in Scopus r Google Scholar r

Meadley et al., 2021 B. Meadley, K.-A. Bowles, K. Smith, L. Perraton, J. Caldwell
Defining the characteristics of physically demanding winch rescue in helicopter search and rescue operations

Appl. Ergon., 93 (2021), Article 103375
View PDF
View article
View in Scopus π
Google Scholar π

Mestria and Engineering, 2018 M.J.C. Mestria, I. Engineering
New Hybrid Heuristic Algorithm for the Clustered Traveling Salesman Problem, vol. 116 (2018), pp. 1-12
View PDF
View article
View in Scopus r
Google Scholar π

Morin et al., 2023 M. Morin, I. Abi-Zeid, C.-G. Quimper
Ant colony optimization for path planning in search and rescue operations
Eur. J. Oper. Res., 305 (2023), pp. 53-63
View PDF
View article
View in Scopus r
Google Scholar π

Qian et al., 2024 J. Qian, Y. Zhou, X. Han, Y. Wang
MDBSCAN: a multi-density DBSCAN based on relative density
Neurocomputing, 576 (2024), Article 127329
View PDF
View article
View in Scopus r
Google Scholar π

Ran et al., 2023 X. Ran, Y. Xi, Y. Lu, X. Wang, Z. Lu
Comprehensive survey on hierarchical clustering algorithms and the recent developments
Artif. Intell. Rev., 56 (2023), pp. 8219-8264
Crossref $\lambda$ View in Scopus $\lambda$ Google Scholar $\lambda$

Randerson et al., 2006 J.T. Randerson, H. Liu, M.G. Flanner, S.D. Chambers, Y. Jin, P.G. Hess, G. Pfister, M. Mack, K. Treseder, L. Welp
The impact of boreal forest fire on climate warming
science, 314 (2006), pp. 1130-1132
Crossref $\lambda$ View in Scopus $\lambda$ Google Scholar $\lambda$

Safford et al., 2022 H.D. Safford, A.K. Paulson, Z.L. Steel, D.J. Young, R.B. Wayman
The 2020 California fire season: a year like no other, a return to the past or a harbinger of the future?

Global Ecol. Biogeogr., 31 (2022), pp. 2005-2025
Crossref $\lambda$ View in Scopus $\lambda$ Google Scholar $\lambda$

Sun et al., 2022 Z.H. Sun, X. Luo, E.Q. Wu, T.Y. Zuo, Z.R. Tang, Z. Zhuang
Monitoring scheduling of drones for emission control areas: an ant colony-based approach
IEEE Trans. Intell. Transport. Syst., 23 (2022), pp. 11699-11709

## Search and rescue under the forest canopy using multiple UAVs

Int. J. Robot Res., 39 (2020), pp. 1201-1221
Crossref π View in Scopus π Google Scholar π

Wang and Han, 2021 Y. Wang, Z. Han
Ant colony optimization for traveling salesman problem based on parameters optimization
Appl. Soft Comput., 107 (2021), Article 107439
View PDF View article View in Scopus $\lambda$ Google Scholar $\lambda$

Wang et al., 2016 X. Wang, T.M. Choi, H. Liu, X. Yue
Ant colony optimization methods for simplifying solution construction in vehicle routing problems
IEEE Trans. Intell. Transport. Syst., 17 (2016), pp. 1-10
View PDF View article Google Scholar π

Wang et al., 2023a H. Wang, Y. Feng, X. Huang, W. Guo
An AI-based action detection UAV system to improve firefighter safety
International Conference on Human-Computer Interaction, Springer (2023), pp. 632-641
Google Scholar π

Wang et al., 2023b Y. Wang, Z. Su, Q. Xu, R. Li, T.H. Luan, P. Wang
A secure and intelligent data sharing scheme for UAV-assisted disaster rescue
IEEE/ACM Transactions on Networking: A Joint Publication of the IEEE Communications Soceity, the IEEE Computer Society, and the ACM with its Special Interest Group on Data Communication, vol. 31 (2023)
Google Scholar π

Xu et al., 2021 X. Xu, L. Zhang, M. Trovati, F. Palmieri, E. Asimakopoulou, O. Johnny, N. Bessis
PERMS: an efficient rescue route planning system in disasters
Appl. Soft Comput., 111 (2021), Article 107667
View PDF View article View in Scopus $\lambda$ Google Scholar $\lambda$
Yang and Li, 2010 S. Yang, C. Li
A clustering particle swarm optimizer for locating and tracking multiple optima in dynamic environments
IEEE Trans. Evol. Comput., 14 (2010), pp. 959-974
View in Scopus $\lambda$ Google Scholar $\lambda$
Yin et al., 2018 C. Yin, Z. Xiao, X. Cao, X. Xi, P. Yang, D. Wu
Offline and online search: UAV multiobjective path planning under dynamic urban environment
IEEE Internet Things J., 5 (2018), pp. 546-558
Crossref $\lambda$ View in Scopus $\lambda$ Google Scholar $\lambda$
Zhang and Zhang, 2017 Q. Zhang, C. Zhang
An improved ant colony optimization algorithm with strengthened pheromone updating mechanism for constraint satisfaction problem
Neural Comput. Appl., 30 (2017), pp. 3209-3220
Google Scholar $\lambda$
Zhang et al., 2020 H. Zhang, Z. Liang, H. Liu, R. Wang, Y. Liu
Ensemble framework by using nature inspired algorithms for the early-stage forest fire rescue a case study of dynamic optimization problems
Eng. Appl. Artif. Intell., 90 (2020)

# Helicopter-UAVs search and rescue task allocation considering UAVs operating environment and performance 

Comput. Ind. Eng., 167 (2022), Article 107994
View PDF View article View in Scopus $\lambda$ Google Scholar $\lambda$
Zhang et al., 2023 C. Zhang, W. Zhou, W. Qin, W. Tang
A novel UAV path planning approach: heuristic crossing search and rescue optimization algorithm
Expert Syst. Appl., 215 (2023), Article 119243
View PDF View article View in Scopus $\lambda$ Google Scholar $\lambda$

Zhao and Wang, 2024 J. Zhao, Y. Wang
Task-driven research on helicopter emergency rescue path planning
Journal of Science, Technology and Society, 3 (2024), p. 10
Google Scholar π

## Cited by (7)

A dynamic adaptive iterative greedy algorithm for collaborative helicopter rescue in post-disaster contexts with finite survival time constraints

2026, Expert Systems with Applications
Show abstract

A decision-making framework by large language model for green tide salvage ship scheduling 2026, Expert Systems with Applications

Show abstract

Real-time quantitative analysis of wildfire fireline merging behavior based on segmentationskeletonization algorithm

2026, Engineering Applications of Artificial Intelligence
Show abstract

Route planning system for mountainous logistics with UAV based on improved algorithm ヶ 2025, Proceedings of SPIE the International Society for Optical Engineering

Review of Thrust Vectoring Technology Applications in Unmanned Aerial Vehicles r 2025, Drones

Just-in-Time Optimal Routing in the Presence of Non-Uniform and Time-Evolving Uncertainty ヶ 2025, Applied Sciences Switzerland

View all citing articles on Scopus
![](https://cdn.mathpix.com/cropped/286e4446-4f36-48bc-9a51-5f151b677c98-33.jpg?height=150&width=141&top_left_y=338&top_left_x=118)

All content on this site: Copyright © 2026 Elsevier B.V., its licensors, and contributors. All rights are reserved, including those for text and data mining, AI training, and similar technologies. For all open access content, the relevant licensing terms apply.
![](https://cdn.mathpix.com/cropped/286e4446-4f36-48bc-9a51-5f151b677c98-33.jpg?height=53&width=193&top_left_y=599&top_left_x=128)

