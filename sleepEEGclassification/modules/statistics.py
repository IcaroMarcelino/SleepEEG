import numpy
from deap import tools

class runtime_metrics:
    def init_stats():
        '''
            Fitness metrics
        '''
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_fit.register("avg", numpy.mean, axis=0, dtype=numpy.float16)
        stats_fit.register("std", numpy.std, axis=0, dtype=numpy.float16)
        stats_fit.register("min", numpy.min, axis=0)
        stats_fit.register("max", numpy.max, axis=0)
        
        '''
            Tree height metrics
        '''
        stats_size = tools.Statistics(lambda ind: ind.height)
        stats_size.register("avg", numpy.mean, dtype=numpy.float16)
        stats_size.register("std", numpy.std, dtype=numpy.float16)
        # stats_size.register("min", numpy.min)
        # stats_size.register("max", numpy.max)

        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        return mstats
