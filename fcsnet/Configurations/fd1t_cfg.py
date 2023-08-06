from Configurations import config as cfg


###############################################################################
# SIMULATION PARAMETERS

# number of simulation parameters
idxSimTypes = len(cfg.EnumSimParams)

# base values of simulation parameters
sim = [0] * idxSimTypes

# acf parameters
correlatorP = 16
correlatorQ = 8
chanum = int(correlatorP + (correlatorQ - 1) * correlatorP/2)
sim[cfg.EnumSimParams.chanum.value] = chanum
