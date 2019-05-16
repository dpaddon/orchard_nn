import numpy as np


class OrchardNN:
    def __init__(self, candidates, distance):
        """
        Args:
            candidates: list of candidates to be assessed
            distance: function to calculate distances between given pairs of points
        """
        
        self.candidates = candidates
        self.distance = distance
        
        self.num_points = len(candidates)
        
        # Precompute all distance pairs
        # e.g. self.precomp_distances[i][j] = distance from i to j
        self.precomp_distances = [[self.distance(x,y) for y in self.candidates] for x in self.candidates]
        
        # For each candidate, compute a list of all other points sorted by distance
        self.ordered_lists = [np.argsort(self.precomp_distances[x]) for x in range(len(candidates))]
        
    def nearest_neighbour(self, query, verbose=True):
        """
        Args: 
            query: the query point
        Returns:
            the nearest neighbour for the query point
        """
        
        # used to avoid repeatedly checking the same point
        tested = [False] * self.num_points
        
        # pick a random point as our best guess
        start_index = np.random.randint(self.num_points)
        
        # say the best index, distance, and corresponding list of distances correspond to this random point
        best_index = start_index
        best_distance = self.distance(query, self.candidates[best_index])
        
        
        i = 0
        while (i < self.num_points): # until we reach the end of the current list
            if best_distance <= (0.5 * self.precomp_distances[best_index][self.ordered_lists[best_index][1]]):
                # This checks if the distance from our query to our best guess is < half
                # the distance from our best guess to its nearest neighbour
                # if so, our best guess is the optimal candidate as all other points must be 
                #further from our query
                break
                
            else:
                # Equally, if we get to the end of the ordered list for the candidate without finding a better
                # distance, then this candidate was the nearest neighbour

                node = self.ordered_lists[best_index][i]

                #this condition avoids unnecessarily checking candidates multiple times
                if tested[node] == False:
                    tested[node] = True
                    
                    query_distance = self.distance(query, self.candidates[node])

                    if query_distance < best_distance:
                        # if True, we've found a nearer neighbour and we assign the index and distance accordingly
                        best_index = node
                        best_distance = query_distance

                        # set i=0 to move to the start of the list of distances for this new best guess
                        i=0

                    else:
                        # look at the next point in the ordered list of neighbours
                        i += 1
                else:
                   #  if we have already tested this point, move on to the next one
                    i += 1

        if verbose:   
            print("Query point: {}".format(query))
            print("Nearest neighbour to query is point {}: {}".format(best_index, self.candidates[best_index]))
            print("Distance: {}".format(best_distance))

        return self.candidates[best_index]
    