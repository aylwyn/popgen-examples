#!/usr/bin/env python3.6
# Aylwyn Scally 2019

import os
import sys
import argparse
import msprime
import tskit
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import logging
from sklearn.decomposition import PCA
from logging import error, warning, info, debug, critical
import random

p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument('-v', '--verbose', action='store_true', default = False)
p.add_argument('--debug', action='store_true', default = False)
p.add_argument('--demdebug', action='store_true', default = False)
p.add_argument('--hc', action='store_true', default = False)
p.add_argument('--seqlen', type=float, default=10e6, help = 'sequence length')
p.add_argument('--mu', type=float, default=1e-8, help = 'mutation rate')
p.add_argument('--rho', type=float, default=2e-8, help = 'recombinaion rate')
p.add_argument('--N0', type=int, default=5000, help = 'initial population size')
p.add_argument('--nsnps', type=int, default=5000, help = 'number of SNPs to use')
p.add_argument('-s', '--nsamps', type = int, default = 4, help = 'number of samples (default = 4)')
p.add_argument('--npops', type = int, default = 2, help = 'number of populations (default = 2)')
p.add_argument('-p', action = 'append', default = [], nargs = '*', help = 'add present-day population: -p samples [Ne [growth_rate]]; (overrides -s, --npops)', metavar = ('samples', 'Ne'))#, 'growth_rate'))
p.add_argument('-j', action='append', default = [], nargs = '*', help='population merge_history', metavar = ('time from_pop to_pop', 'proportion'))
p.add_argument('-m', action='append', default = [], nargs = '*', help='migration matrix element (add s for symmetric)', metavar = ('from_pop to_pop to_pop_frac', 's'))
#p.add_argument('--em', action='append', default = [], nargs = 4, help='migration_history', metavar = ('time', 'to_pop', 'from_pop', 'to_pop_mig_frac'))
p.add_argument('--mglobal', type=float, default=0.0, help = 'global symmetric migration rate')
p.add_argument('--prune_by_sep', type=float, default=0.0, help = 'min separation between SNPs (otherwise use LD)')
p.add_argument('--LD_threshold', type=float, default=0.2, help = 'max LD between SNPs')
p.add_argument('--seed', type=int, default=0, help = 'random seed for simulation')
p.add_argument('--pcasamples', default='', help = 'number of samples to include in PCA from each population: n1,n2,...')
p.add_argument('--project_all', action='store_true', default = False, help = 'project all samples onto the PC axes')
p.add_argument('--figtitle', default='', help = 'figure title')
p.add_argument('--figfmt', default='svg', choices=['svg', 'pdf', 'png'], help = 'figure format')
p.add_argument('--figsize', default = '5.0,5.0', help = 'figure size in inches')
args = p.parse_args()

loglevel = logging.WARNING
if args.verbose:
	loglevel = logging.INFO
if args.debug:
	loglevel = logging.DEBUG
logging.basicConfig(format = '%(module)s:%(lineno)d:%(levelname)s: %(message)s', level = loglevel)

fsize = [float(x) for x in args.figsize.split(',')]
fig, ax = plt.subplots(1, 1)
plt.sca(ax)

encmd = []
popconfig = []
demevents = []
samppop = []
popsize = []

if not args.p: # equal pop sizes
	roundpop = round(args.nsamps/args.npops)
	if not roundpop * args.npops == args.nsamps:
		warning('nsamps %d not a multiple of npops %d; setting nsamps = %d' % (args.nsamps, args.npops, roundpop * args.npops))
		args.nsamps = roundpop * args.npops
	popsize = [roundpop for pop in range(args.npops)]
	if args.npops > 1:
		info('distributing %d samples among %d populations' %  (args.nsamps, args.npops))
	for ip in range(args.npops):
		popconfig.append(msprime.PopulationConfiguration(sample_size=popsize[ip], initial_size=args.N0))
		samppop += [ip] * roundpop
#		encmd.append('-p %d' % roundpop)
	encmd.append('%dx%d' % (args.npops, roundpop))
else:
	args.nsamps = 0
	for ix, x in enumerate(args.p):
		args.nsamps += int(x[0])
		samppop += [ix] * int(x[0])
		popsize.append(int(x[0]))
		if len(x) == 1:
			popconfig.append(msprime.PopulationConfiguration(sample_size=int(x[0]), initial_size=args.N0))
		elif len(x) == 2:
			popconfig.append(msprime.PopulationConfiguration(sample_size=int(x[0]), initial_size=eval(x[1])))
		elif len(x) == 3:
			popconfig.append(msprime.PopulationConfiguration(sample_size=int(x[0]), initial_size=eval(x[1]), growth_rate=eval(x[2])))
		else:
			error('max 3 arguments for -p: samples [initial_Ne [growth_rate]]')
			sys.exit(1)
		encmd.append('-p ' + '_'.join(x))
	args.npops = len(popconfig)
info(f'population sample distribution: {popsize}')

if args.pcasamples:
	pcasamples = [int(x) for x in args.pcasamples.split(',')]
	if len(pcasamples) != args.npops:
		error('length of pcasamples differs from number of populations')
		sys.exit()

migmatrix = np.zeros((args.npops, args.npops))
for i in range(args.npops):
	for j in range(args.npops):
		if i != j:
			migmatrix[i][j] = args.mglobal

for ix, x in enumerate(args.m):
	if len(x) == 3:
		migmatrix[int(x[0])][int(x[1])] = eval(x[2])
	elif len(x) == 4 and x[3] == 's':
		migmatrix[int(x[0])][int(x[1])] = eval(x[2])
		migmatrix[int(x[1])][int(x[0])] = eval(x[2])
	else:
		error('usage for -m: from_pop to_pop mig_rate [s]')
		sys.exit(1)
	encmd.append('-m ' + '_'.join(x))


for ix, x in enumerate(args.j):
	if len(x) == 3:
		demevents.append(msprime.MassMigration(time=eval(x[0]), source = int(x[1]), destination = int(x[2]), proportion=1.0))
	elif len(x) == 4:
		demevents.append(msprime.MassMigration(time=eval(x[0]), source = int(x[1]), destination = int(x[2]), proportion=float(x[3])))
	else:
		error('3 or 4 arguments for --j: time from_pop to_pop [proportion]')
		sys.exit(1)
	encmd.append('-j ' + '_'.join(x))

# show the demographic history
dd = msprime.DemographyDebugger(population_configurations=popconfig, migration_matrix=migmatrix, demographic_events=demevents)
dd.print_history()
if args.demdebug:
	sys.exit()

info('simulating..')
if not args.seed:
	args.seed = round(1000 * random.random())
	info(f'using random seed {args.seed}')

ts = msprime.simulate(population_configurations=popconfig, migration_matrix=migmatrix, demographic_events=demevents, length=args.seqlen, mutation_rate=args.mu, recombination_rate=args.rho, random_seed=args.seed)

variants = np.empty((args.nsnps, args.nsamps), dtype="u1")
j = 0
if args.prune_by_sep > 0:
	info(f'pruning sites by separation > {args.prune_by_sep}')
	lastpos = 0
	for var in ts.variants():
		if lastpos + args.prune_by_sep > args.seqlen:
			error(f'ran out of sequence after {j} SNPs')
			break
		if var.site.position - lastpos > args.prune_by_sep:
	#		debug(j, var.site.id, var.site.position, var.genotypes, sep="\t")
			variants[j] = var.genotypes
			j += 1
			lastpos = var.site.position
		if j == args.nsnps:
			break
else:
	info(f'LD pruning with threshold {args.LD_threshold}')
	ld = tskit.LdCalculator(ts)
	next_mutn = 0
	tvars = ts.variants()
	var = next(tvars)
	while j < args.nsnps:
		r2 = (ld.r2_array(next_mutn, max_mutations=50) < args.LD_threshold)
		if len(r2) == 0 or not np.any(r2):
			error(f'ran out of sites after {j} mutations below LD theshold')
			break
		next_mutn += (1 + np.argmax(r2))
		while var.site.id < next_mutn:
			var = next(tvars)
		variants[j] = var.genotypes
		debug(f'SNP {j}: {var.site.id} at {var.site.position}')
		j += 1
	info(f'last SNP {j}: {var.site.id} at {var.site.position}')

outargs = ' '.join(encmd)
outname = 'pca' + '_'.join(outargs.split()).replace('_-', '-')

info(f'running PCA')
data = np.ndarray.transpose(variants)
debug(f'data array: {data.shape}')

if args.pcasamples: # use subset of samples to define PCs
	popindex = 0
	dataslice = []
	for ip in range(args.npops):
		dataslice += range(popindex, popindex + pcasamples[ip])
		popindex += popsize[ip]
	pcadata = data[dataslice]
	samppopc = [samppop[i] for i in dataslice]
	debug(f'using samples {dataslice}')

pca = PCA(n_components=2)
pca.fit(pcadata)
#PCA(copy=True, iterated_power='auto', n_components=2, random_state=None, svd_solver='full', tol=0.0, whiten=False)

if args.project_all:
	samps_pca = pca.transform(data)
else:
	samps_pca = pca.transform(pcadata)
	samppop = samppopc

colarr = ['#ee8800', '#ee00ff', '#0033ff', '#00ef00', '#ee0000']

plt.scatter(samps_pca[:, 0], samps_pca[:, 1], color=[colarr[x] for x in samppop])
plt.xlabel(f'PC1 ({100*pca.explained_variance_ratio_[0]:.2f} %)')
plt.ylabel(f'PC2 ({100*pca.explained_variance_ratio_[1]:.2f} %)')

if args.figtitle:
	plt.title(args.figtitle)
	plt.subplots_adjust(left=0.15, bottom=0.1, right=1 - 0.05, top= 1 - 0.075)
else:
	plt.subplots_adjust(left=0.15, bottom=0.1, right=1 - 0.05, top= 1 - 0.05)

if args.hc:
	fig.set_size_inches(fsize[0], fsize[1])
	fname = '.'.join([outname + '_' + args.figtitle, args.figfmt])
	plt.savefig(fname, format=args.figfmt)
	info(f'Saved figure to {fname}')
else:
	plt.show()
