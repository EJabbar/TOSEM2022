import imp
from get_coverage import get_coverage
from LC_tp import LC
from BC_tp import BC

p_name = 'Time'
for v in range(23, 28):
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>> ', v)
    # get_coverage(p_name, v) #uncomment this for cloning the projects
    LC(p_name, v)
    BC(p_name, v)
