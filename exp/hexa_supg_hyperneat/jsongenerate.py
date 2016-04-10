#!/usr/bin/python

# Python script to generate json files to continue evolution experiment with clone variant populations

expname = 'hexa_supg_hyperneat'
bindir = '/home/tarapore/svn_isir/sferes2/trunk/build/default/exp/hexa_supg_hyperneat' # where is you executable
resdir = '/home/tarapore/exp/hexa_supg_hyperneat/lmBr/exp_' # where are the results to be stored

#################################################
# Arguments for seeded NSGA2
genfile = '/home/tarapore/exp/hexa_supg_hyperneat/exp_'  # where are the generation files
generation = 8000                                       # generation to load 
indid = 0                                               # id of individual to clone and mutate, seeding the population 
#################################################


for trial in range(0, 20):
        fn=expname + str(trial) + '.json'
        f = open(fn, 'w')
        f.write('{\n')
        f.write('   \"email\" : \"daneshtarapore@gmail.com\",\n')
        f.write('   \"wall_time" : \"11:59:59\",\n')
        f.write('   \"nb_runs" : 1,\n')

        bin_dir = '   \"bin_dir\": \"' + bindir + '\",\n'
        f.write(bin_dir)


        res_dir = '   \"res_dir\": \"' + resdir + str(trial) + '\",\n'
        f.write(res_dir)

        exp_name = '   \"exps\" : [\"' + expname + '_text' + '\"],\n'
        f.write(exp_name)


        args = '   \"args\" : \"' + genfile + str(trial) + '/node*/gen_' + str(generation)  + ' ' + str(indid) + '\"\n'
        f.write(args)

        f.write('}\n')
        f.close()
