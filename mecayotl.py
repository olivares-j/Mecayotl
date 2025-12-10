# Copyright 2023 Javier Olivares Romero
#
# This file is part of Mecayotl.
#
# 	Mecayotl is free software: you can redistribute it and/or modify
# 	it under the terms of the GNU General Public License as published by
# 	the Free Software Foundation, either version 3 of the License, or
# 	(at your option) any later version.
#
# 	Mecayotl is distributed in the hope that it will be useful,
# 	but WITHOUT ANY WARRANTY; without even the implied warranty of
# 	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# 	GNU General Public License for more details.
#
# 	You should have received a copy of the GNU General Public License
# 	along with Mecayotl.  If not, see <http://www.gnu.org/licenses/>.
#
# NOTE: This copy of the file has been augmented *only* with explanatory
# comments to help readability and understanding. No logic, variable names,
# or behavior has been changed.
#
# High-level overview:
# - Mecayotl is a coordinating class that drives data assembly, GMM inference,
#   probability computation and synthetic experiments to identify cluster
#   members using external packages (Amasijo, Kalkayotl, a local GMM).
# - Many functions read/write HDF5 datasets, call external inference code,
#   and produce plots. Careful attention is required when changing I/O paths.
#
# Comments have been added to explain purpose of modules, major steps inside
# functions, and to annotate non-obvious operations (e.g., covariance assembly).
#
# For contributors: prefer reading the suggestions in the README or the
# separate suggestions section (outside this file) before attempting to
# refactor or rework this implementation.

import sys
import os
import numpy as np
import pandas as pd
import h5py
import dill

from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from astropy.table import Table
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord, search_around_sky,FK5
from astropy import units

# PyGaia helpers for astrometry conversions
from pygaia.astrometry.vectorastrometry import phase_space_to_astrometry
from pygaia.astrometry.constants import au_km_year_per_sec,au_mas_parsec

# Matplotlib backend chosen to be non-interactive (suitable for servers)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm
from matplotlib import lines as mlines
from matplotlib.patches import Ellipse
from matplotlib.colors import Normalize,TwoSlopeNorm
from tqdm import tqdm

#-------------- Amasijo libraries (external) --------
from Amasijo import Amasijo
from Amasijo.Quality import ClassifierQuality
#-----------------------------------------

# Configure Simbad query to include velocity fields when used.
# (This affects astroquery.Simbad global settings)
Simbad.add_votable_fields('velocity')

def get_principal(sigma,level=2.0):
	"""
	Compute principal axes lengths and orientation for a 2x2 covariance matrix.

	Parameters
	----------
	sigma : 2x2 numpy.array
		The covariance matrix (for two dimensions).
	level : float
		Scaling factor for the returned ellipse axes (e.g., 2 -> ~95% contour).

	Returns
	-------
	width, height, angle_deg
		Width and height (semi-axes lengths scaled by 'level') and angle in degrees.
	"""
	sd_x   = np.sqrt(sigma[0,0])
	sd_y   = np.sqrt(sigma[1,1])
	rho_xy = sigma[0,1]/(sd_x*sd_y)


	# Author: Jake VanderPlas
	# License: BSD
	# The computation below extracts eigenvalue-like quantities without
	# explicitly calling eig to keep things compact.
	sigma_xy2 = rho_xy * sd_x * sd_y

	alpha = 0.5 * np.arctan2(2 * sigma_xy2,(sd_x ** 2 - sd_y ** 2))
	tmp1  = 0.5 * (sd_x ** 2 + sd_y ** 2)
	tmp2  = np.sqrt(0.25 * (sd_x ** 2 - sd_y ** 2) ** 2 + sigma_xy2 ** 2)

	# Returns major axis, minor axis and rotation angle in degrees.
	return level*np.sqrt(tmp1 + tmp2), level*np.sqrt(np.abs(tmp1 - tmp2)), alpha* 180. / np.pi

class Mecayotl(object):
	"""
	Mecayotl is an algorithm designed to identify members of open clusters.

	High-level responsibilities:
	- Manage configuration, file paths and default arguments.
	- Interface with Amasijo to generate synthetic clusters.
	- Assemble real Gaia+members data into HDF5 for modeling.
	- Fit Gaussian Mixture Models (GMM) for field and cluster with a
	  local GMM implementation (gmm.GaussianMixture).
	- Compute cluster membership probabilities.
	- Run synthetic experiments and evaluate classifier thresholds using
	  Amasijo synthetic stars and provided metrics.

	Note: Many parameters are passed as dicts so that defaults can be filled.
	"""
	def __init__(self,dir_base,
		file_members,
		file_gaia,
		members_args={},
		isochrones_args={},
		kalkayotl_args={},
		nc_cluster=range(2,21),
		nc_field=range(2,21),
		path_ayome     = "/home/jolivares/Repos/Ayome_GPU/",
		path_kalkayotl = "/home/jolivares/Repos/Kalkayotl/",
		cmap_probability="viridis_r",
		cmap_features="viridis_r",
		zero_points={
					"ra":0.,
					"dec":0.,
					"parallax":-0.017,# This is Brown+2020 value
					"pmra":0.,
					"pmdec":0.,
					"radial_velocity":0.
					},
		observable_limits={},
		mapper_names={
		"radial_velocity":"dr3_radial_velocity",
		"phot_g_mean_mag":"g",
		"phot_bp_mean_mag":"bp",
		"phot_rp_mean_mag":"rp"},
		reference_system="Galactic",
		model="Gaussian",
		seed=1234):

		# Base directory where outputs/iter_* will be created.
		self.dir_base = dir_base

		# Gaia observable column ordering used across the pipeline.
		# The order matters because code later slices arrays expecting this order.
		gaia_observables = ["source_id",
		"ra","dec","parallax","pmra","pmdec","radial_velocity",
		"ra_error","dec_error","parallax_error","pmra_error","pmdec_error","radial_velocity_error",
		"ra_dec_corr","ra_parallax_corr","ra_pmra_corr","ra_pmdec_corr",
		"dec_parallax_corr","dec_pmra_corr","dec_pmdec_corr",
		"parallax_pmra_corr","parallax_pmdec_corr",
		"pmra_pmdec_corr",
		"phot_g_mean_mag",
		"phot_bp_mean_mag",
		"phot_rp_mean_mag",
		"ruwe"]

		#--------------- Observable limits -------------------
		# Fill missing observable limits with safe defaults.
		default_observable_limits = {
			"ra":{"inf":-np.inf,"sup":np.inf},
			"dec":{"inf":-np.inf,"sup":np.inf},
			"pmra":{"inf":-400.0,"sup":400.0},
			"pmdec":{"inf":-400.0,"sup":400.0},
			"parallax":{"inf":-100,"sup":100},
			"radial_velocity":{"inf":-200.0,"sup":200.0}
			}
		for arg,val in default_observable_limits.items():
			if not arg in observable_limits:
				observable_limits[arg] = val

		self.observable_limits = observable_limits
		print("The following observable limits will be used:")
		for k,v in self.observable_limits.items():
			print("{0} : {1}".format(k,v))

		#---------------- Members arguments default and fill --------------
		members_default_args = {
		"g_mag_limit":22.0,
		"rv_error_limits":[0.1,2.0],
		"ruwe_threshold":1.4,
		"prob_threshold":0.999936,
		"rv_sd_clipping":3.0,
		"allow_rv_missing":False,
		}

		for arg,val in members_default_args.items():
			if not arg in members_args:
				members_args[arg] = val

		self.members_args = members_args
		print("The following members_args will be used:")
		for k,v in self.members_args.items():
			print("{0} : {1}".format(k,v))
		#----------------------------------------------------------

		#---------------- Kalkayotl arguments default and fill ----------
		# These are defaults for the Bayesian inference wrapper used in 'clean_members' and 'run_kalkayotl'.
		kalkayotl_default_args = {
		"distribution":"Gaussian",
		"statistic":"mean",
		"tuning_iters":2000,
		"sample_iters":1000,
		"target_accept":0.65,
		"chains":2,
		"cores":2,
		"step":None,
		"step_size":1e-3,
		"init_method":"advi+adapt_diag",
		"init_iters":int(1e6),
		"init_absolute_tol":5e-3,
		"init_relative_tol":1e-5,
		"init_plot_iters":int(1e4),
		"init_refine":False,
		"prior_predictive":False,
		"nuts_sampler":"pymc",
		"random_seed":None,
		"parameterization":"central",
		"velocity_model":"joint",
		"min_gmm_components":2,
		"max_gmm_components":2,
		"hdi_prob":0.95,
		"sampling_space":"physical",
		"sky_error_factor":1e6
		}

		for arg,val in kalkayotl_default_args.items():
			if not arg in kalkayotl_args:
				kalkayotl_args[arg] = val

		self.kalkayotl_args = kalkayotl_args
		print("The following kalkayotl_args will be used:")
		for k,v in self.kalkayotl_args.items():
			print("{0} : {1}".format(k,v))
		#----------------------------------------------------------

		#---------------- Isochrones arguments default and fill -------
		isochrones_default_args = {
		"log_age": 8.0,
		"metallicity":0.012,
		"Av": 0.0,
		"mass_limits":[0.1,2.0],
		"bands":["G","BP","RP"],
		"mass_prior":"Uniform"
		}

		for arg,val in isochrones_default_args.items():
			if not arg in isochrones_args:
				isochrones_args[arg] = val

		self.isochrones_args = isochrones_args
		print("The following isochrones_args will be used:")
		for k,v in self.isochrones_args.items():
			print("{0} : {1}".format(k,v))
		#----------------------------------------------------------

		#-------------------------------- Mappers ----------------------------------------------
		# mapper_names maps canonical names used in this code to the column names found in the input data.
		col_names = ["radial_velocity","phot_g_mean_mag","phot_rp_mean_mag","phot_bp_mean_mag"]
		assert set(col_names).issubset(set(mapper_names.keys())),\
		"Error: the following column names must be present in the mapper:\n"+col_names

		# output_mapper: map from canonical to input (column names in provided files)
		output_mapper = mapper_names
		# input_mapper: inverse mapper (from input file column to canonical)
		input_mapper =  { v:k for k, v in mapper_names.items()}

		self.input_mapper = input_mapper.copy()
		self.output_mapper = output_mapper.copy()

		# Add _error variants to the mapper automatically
		base_error = "{0}_error"
		for k,v in input_mapper.items():
			self.input_mapper[base_error.format(k)] = base_error.format(v)
		for k,v in output_mapper.items():
			self.output_mapper[base_error.format(k)] = base_error.format(v)
		#----------------------------------------------------------------------------------------

		#------ Set seed for reproducibility -----------------
		np.random.seed(seed=seed)
		self.random_state = seed
		self.seed = seed
		#----------------------------------------------------

		#------------- Repo paths -----------------
		self.path_ayome      = path_ayome
		self.path_kalkayotl  = path_kalkayotl
		#-------------------------------------------------

		#------------- Files  -------------------------
		# Paths to the raw Gaia catalogue and initial members list
		self.file_gaia    = file_gaia
		self.file_members = file_members
		#-------------------------------------

		#-------------- Parameters -----------------------------------------------------
		self.zero_points = zero_points
		self.cmap_prob = plt.get_cmap(cmap_probability)
		self.cmap_feat = plt.get_cmap(cmap_features)
		# idxs: index pairs for plotting different projections of the 6D space
		self.idxs      = [[0,1],[2,1],[0,2],[3,4],[5,4],[3,5]]
		# plots: named pairs used when plotting members & candidates
		self.plots     = [
						  ["ra","dec"],["pmra","pmdec"],
						  ["parallax","pmdec"],["g_rp","phot_g_mean_mag"],["g_rp","G"]
						 ]
		self.IDS       = gaia_observables[0]
		self.OBS       = gaia_observables[1:7]
		self.UNC       = gaia_observables[7:13]
		self.RHO       = gaia_observables[13:23]
		self.EXT       = gaia_observables[23:]
		self.PRO       = "prob_cls"
		self.nc_case   = {"Field":nc_field,"Cluster":nc_cluster}
		self.best_gmm  = {}
		self.isochrones_args = isochrones_args
		self.observables = gaia_observables
		self.reference_system = reference_system

		#----------- APOGEE specific renaming where applicable ---------------
		self.apogee_columns = ["RA","DEC","GAIAEDR3_SOURCE_ID","VHELIO_AVG","VSCATTER","VERR"]
		self.apogee_rename = {"VHELIO_AVG":"apogee_rv","GAIAEDR3_SOURCE_ID":"source_id"}
		#------------------------------------------------------------------------

		#----- Kalkayotl: add path & import its Inference class --------------
		sys.path.append(self.path_kalkayotl)
		from kalkayotl.inference import Inference
		self.Inference = Inference
		#-----------------------------------------

		#============================== Ayome ================================
		# The code expects a generated Python file under path_ayome that contains
		# auxiliary functions for different dimensionalities. The sed call replaces
		# a placeholder DIMENSION with 6 in the base file and writes Functions.py.
		cmd = 'sed -e "s|DIMENSION|{0}|g"'.format(6)
		cmd += ' {0}Functions_base.py > {0}Functions.py'.format(self.path_ayome)
		os.system(cmd)
		sys.path.append(self.path_ayome)
		#---------------------------------------------------------------------

		#-------GaussianMixtureModel ------------------
		# Local GMM implementation imported from gmm.py in repository
		from gmm import GaussianMixture
		self.GMM = GaussianMixture
		#---------------------------------------
		#=======================================================================

	def initialize_directories(self,dir_main):
		"""
		Create and set commonly used file paths for a given run directory.
		All names are stored as attributes for later use by other methods.
		"""
		#--------- Directories ---------------------------
		self.dir_main  = dir_main
		#------------------------------------------

		#---------------- Files --------------------------------------------------
		# Compose common paths used by methods that produce/read data in dir_main.
		self.file_flt_mem    = dir_main + "/Kalkayotl/filtered_members.csv"
		self.file_cln_mem    = dir_main + "/Kalkayotl/clean_members.csv"
		self.file_par_kal    = dir_main + "/Kalkayotl/{0}/Cluster_statistics.csv".format(
			self.kalkayotl_args["distribution"])
		self.file_smp_base   = dir_main + "/{0}/Data/members_synthetic.csv"
		self.file_data_base  = dir_main + "/{0}/Data/data.h5"
		self.file_model_base = dir_main + "/{0}/Models/{1}_GMM_{2}.h5"
		self.file_comparison = dir_main + "/{0}/Models/{1}_comparison.png"
		self.file_qlt_base   = dir_main + "/Classification/quality_{0}_{1}.{2}"
		self.file_mem_data   = dir_main + "/Classification/members_mecayotl.csv"
		self.file_mem_plot   = dir_main + "/Classification/members_mecayotl.pdf"
		#-------------------------------------------------------------------------

		#----- Create directories if missing -----
		os.makedirs(dir_main + "/Real",exist_ok=True)
		os.makedirs(dir_main + "/Real/Data",exist_ok=True)
		os.makedirs(dir_main + "/Real/Models",exist_ok=True)
		os.makedirs(dir_main + "/Kalkayotl",exist_ok=True)
		#-------------------------------------------

	def generate_true_cluster(self,file_kalkayotl,n_cluster=int(1e5),instance="Real"):
		"""
		Generate synthetic data based on Kalkayotl input parameters.

		Uses Amasijo to generate phase-space positions and convert them to
		observables (mu = mean observables) that match Gaia-like columns.
		Results are saved to CSV file defined by file_smp_base for the instance.
		"""
		file_smp  = self.file_smp_base.format(instance)

		#----------- Generate true astrometry through Amasijo ---------------
		ama = Amasijo(kalkayotl_args={
						"file":file_kalkayotl,
						"statistic":self.kalkayotl_args["statistic"]},
					  isochrones_args=self.isochrones_args,
					  reference_system=self.reference_system,
					  radial_velocity={
					  	"labels":{"radial_velocity":"dr3_radial_velocity"},
					  	"family":"Gaia"},
					  seed=self.seed)

		# X is raw phase-space output: Amasijo internal representation.
		X = ama._generate_phase_space(n_stars=n_cluster)

		# Convert to observable astrometry (parallax, PMs, RV etc.)
		df_syn,_ = ama._generate_true_astrometry(X)
		#----------------------------------------------------

		#----- Rename Amasijo output columns to the observables expected by this pipeline.
		df_syn.rename(columns={
			ama.labels_true_as[0]:self.observables[1],#"ra",
			ama.labels_true_as[1]:self.observables[2],#"dec",
			ama.labels_true_as[2]:self.observables[3],#"parallax",
			ama.labels_true_as[3]:self.observables[4],#"pmra",
			ama.labels_true_as[4]:self.observables[5],#"pmdec",
			ama.labels_true_as[5]:self.observables[6],#"radial_velocity"
			},inplace=True)
		#------------------------------------------------

		# Filter synthetic stars using observable_limits
		valid_syn = np.full(len(df_syn),fill_value=True)
		for obs in self.OBS:
			valid_syn &= df_syn[obs] > self.observable_limits[obs]["inf"]
			valid_syn &= df_syn[obs] < self.observable_limits[obs]["sup"]

		df_syn = df_syn.loc[valid_syn]

		# Save synthetic members to CSV (index label source_id to match other code)
		df_syn.to_csv(file_smp,index_label="source_id")

	def assemble_data(self,file_catalogue,file_members,
					n_field=int(1e5),
					instance="Real"):
		"""
		Assemble a dataset combining an input Gaia catalogue and a member list:
		- Read catalogue (FITS) and members (CSV or FITS).
		- Filter catalogue by configured observable limits.
		- Read synthetic cluster sample previously generated by generate_true_cluster.
		- Randomly select n_field star samples from catalogue to represent field.
		- Save arrays and covariance matrices to an HDF5 file used by later steps.
		"""
		#------------ Files ------------------------------
		file_data = self.file_data_base.format(instance)
		file_smp  = self.file_smp_base.format(instance)
		#-------------------------------------------------

		#--------------- Catalogue ---------------------
		print("Reading catalogue ...")
		cat = Table.read(file_catalogue, format='fits')
		df_cat = cat.to_pandas()
		# Rename input columns according to mapper to canonical observable names
		df_cat = df_cat.rename(columns=self.input_mapper).loc[:,self.observables]

		# Apply observable limits allowing NaN values to pass through (they will be handled later).
		valid_cat = np.full(len(df_cat),fill_value=True)
		for obs in self.OBS:
			valid_cat &= (df_cat[obs] > self.observable_limits[obs]["inf"]) |\
						 (np.isnan(df_cat[obs]))
			valid_cat &= (df_cat[obs] < self.observable_limits[obs]["sup"]) |\
						 (np.isnan(df_cat[obs]))

		df_cat = df_cat.loc[valid_cat]
		print("There are {0} valid catalogue sources".format(len(df_cat)))
		n_sources = df_cat.shape[0]
		del cat
		#-----------------------------------------------

		#--------- Members file (only IDs needed here) ------------------------------------------
		print("Reading members ...")
		if '.csv' in file_members:
			# Only source_id column is read (self.IDS)
			df_mem = pd.read_csv(file_members,usecols=[self.IDS])
		elif ".fits" in file_members:
			dat = Table.read(file_members, format='fits')
			df_mem  = dat.to_pandas().loc[:,[self.IDS]]
			del dat
		else:
			sys.exit("Format file not recognized. Only CSV of FITS")
		#-------------------------------------------------------------

		#----------- Synthetic cluster sample -------------------------------------------
		print("Reading synthetic ...")
		df_syn = pd.read_csv(file_smp,usecols=self.OBS)
		valid_syn = np.full(len(df_syn),fill_value=True)
		for obs in self.OBS:
			valid_syn &= df_syn[obs] > self.observable_limits[obs]["inf"]
			valid_syn &= df_syn[obs] < self.observable_limits[obs]["sup"]

		print("There are {0} valid synthetic sources".format(len(valid_syn)))
		# Randomly choose n_field synthetic samples to represent the field distribution.
		idx_rnd  = np.random.choice(len(valid_syn),size=n_field,replace=False)
		mu_syn = df_syn.loc[valid_syn].iloc[idx_rnd]
		# sg_syn is synthetic covariance placeholder (zeros here, replaced later if needed)
		sg_syn = np.zeros((len(mu_syn),6,6))
		del df_syn
		#-----------------------------------------------------------------

		#---- Subtract zero point (instrumental or catalog offsets)
		for obs in self.OBS:
			df_cat[obs] -= self.zero_points[obs]
		#---------------------------------------

		#-------- Sky uncertainties conversion: mas -> degrees for RA/Dec errors
		for obs in ["ra_error","dec_error"]:
			df_cat[obs] *= 1./(60.*60.*3600.)
		#------------------------------------

		print("Assembling data ...")

		#------ Extract arrays that will be saved to HDF5 ---------------------------
		id_data = df_cat.loc[:,self.IDS].to_numpy()
		mu_data = df_cat.loc[:,self.OBS].to_numpy()
		sd_data = df_cat.loc[:,self.UNC].to_numpy()
		cr_data = df_cat.loc[:,self.RHO].to_numpy()
		ex_data = df_cat.loc[:,self.EXT].to_numpy()
		#------------------------------------------

		#----- Select members and field indices based on supplied members file ----------
		ids_all  = df_cat[self.IDS].to_numpy()
		ids_mem  = df_mem[self.IDS].to_numpy()
		mask_mem = np.isin(ids_all,ids_mem)
		idx_cls  = np.where(mask_mem)[0]
		idx_fld  = np.where(~mask_mem)[0]
		#-----------------------------------------

		#-------------- Members sanity check -----------------------------
		assert len(idx_cls) > 1, "Error: Empty members file!"
		#----------------------------------------------------

		#---------- Random sample of field sources ------------------
		msg_err_n_fld = "Error: The requested field number is larger than the filed population ({0} > {1})".format(n_field,len(idx_fld))
		assert len(idx_fld)>= n_field, msg_err_n_fld
		idx_rnd  = np.random.choice(idx_fld,size=n_field,replace=False)
		mu_field = mu_data[idx_rnd]
		#------------------------------------------------------------

		#------------- Write partial data to HDF5 -----------------------------------------
		print("Saving data ...")
		with h5py.File(file_data, 'w') as hf:
			# Basic arrays (ids, means, errors, correlations, etc.)
			hf.create_dataset('ids',        data=id_data)
			hf.create_dataset('mu',         data=mu_data)
			hf.create_dataset('sd',         data=sd_data)
			hf.create_dataset('cr',         data=cr_data)
			hf.create_dataset('ex',         data=ex_data)
			# Synthetic cluster and field placeholders
			hf.create_dataset('mu_Cluster', data=mu_syn)
			hf.create_dataset('sg_Cluster', data=sg_syn)
			hf.create_dataset('mu_Field',   data=mu_field)
			hf.create_dataset('idx_Field',  data=idx_fld)
			hf.create_dataset('idx_Cluster',data=idx_cls)
		#-------------------------------------------------------------

		#-------- Clear memory for large objects -------------------------
		del df_cat,mu_data,mu_field,mu_syn,sg_syn
		#-----------------------------------------------

		#============= Covariance matrices assembly =====================
		# Build full covariance 6x6 arrays for each source using reported uncertainties and correlation coefficients.
		print("Filling covariance matrices ...")
		sg_data = np.zeros((n_sources,6,6))

		#----- There is no correlation with radial_velocity (index 5), so only fill correlations for first 5 dims
		idx_tru = np.triu_indices(6,k=1)
		idi     = np.where(idx_tru[1] != 5)[0]
		idx_tru = (idx_tru[0][idi],idx_tru[1][idi])
		#--------------------------------------------

		#-------- Prepare a diagonal matrix structure with sd_data values
		stds = np.zeros((n_sources,6,6))
		diag = np.einsum('...jj->...j',stds)
		diag[:] = sd_data
		# Note: diag now contains per-source diagonal elements (not squared).
		#--------------------------------------

		one = np.eye(6)
		# Progress bar to show progress for large catalog sizes
		pbar = tqdm(total=n_sources,miniters=10000)
		for i,(sd,corr) in enumerate(zip(stds,cr_data)):
			#------- Build correlation matrix (rho) for indices excluding r_vel ---
			rho  = np.zeros((6,6))
			rho[idx_tru] = corr
			# Make matrix symmetric and set diagonal to 1
			rho  = rho + rho.T + one
			#-----------------------------

			#---------- Covariance = D * rho * D where D = diag(sd) --------------
			sg_data[i] = sd.dot(rho.dot(sd))
			#--------------------------------------------

			pbar.update()
		pbar.close()
		#========================================================

		#----- Field sources covariance subset ------
		sg_field = sg_data[idx_rnd]
		#--------------------------

		#---------------- Write covariance matrices back to the HDF5 file ---------------------------------------
		print("Saving covariance matrices ...")
		with h5py.File(file_data, 'a') as hf:
			hf.create_dataset('sg',      data=sg_data)
			hf.create_dataset('sg_Field',data=sg_field)
		#--------------------------------------------------------------
		del sg_data,sg_field
		print("Data correctly assembled")

	def infer_models(self,case="Field",instance="Real",
					tolerance=1e-5,init_min_det=1e-3):
		"""
		Fit Gaussian Mixture Models (GMM) to either the Field or Cluster
		data stored in HDF5 for a given instance.

		Behavior:
		- If an existing model file for the maximum components exists, it's used
		  to seed the fitting of smaller models.
		- If not found, fit a GMM with max_n_components and save parameters.
		- Then loop over the requested n_components and fit each model, saving it.
		"""
		file_data = self.file_data_base.format(instance)

		#------------ Read data for the requested case ------------------
		print("Reading data ...")
		with h5py.File(file_data, 'r') as hf:
			# X: means for the case (mu_Field or mu_Cluster)
			X = np.array(hf.get("mu_" + case))
			# U: uncertainties (covariances)
			U = np.array(hf.get("sg_" + case))
		#----------------------------------------

		#-- Dimensions --
		N,_ = X.shape
		#----------------

		#------------ Maximum number of components (to seed with) -----
		max_n_components = np.max(self.nc_case[case])
		#-----------------------------------------------

		#============================== Read or infer base initialization =======================================
		file_model = self.file_model_base.format(instance,case,max_n_components)
		init_params = {}

		if os.path.isfile(file_model):
			# If a precomputed model exists, read weights, means, covs and determinants
			with h5py.File(file_model, 'r') as hf:
				init_params["weights"] = np.array(hf.get('pros'))
				init_params["means"]   = np.array(hf.get('means'))
				init_params["covs"]    = np.array(hf.get('covs'))
				init_params["dets"]    = np.array(hf.get('dets'))
		else:
			#+++++++++++++++++++++++++ Inference for the max components +++++++++++++++++++++++++++++++++++++
			print("--------------------------------------------")

			#--------------- Do inference (fit GMM with max components) ----------------------------------------
			print("Inferring model with {0} components.".format(max_n_components))
			gmm = self.GMM(dimension=6,n_components=max_n_components)
			gmm.setup(X,uncertainty=U)
			# The local GMM's fit will handle initialization and optimization.
			gmm.fit(tol=tolerance,
				init_min_det=init_min_det,
				init_params="GMM",
				random_state=self.random_state)
			#--------------------------------------------------------------------

			#------- Write computed model to HDF5 for future reuse ----------------
			with h5py.File(file_model,'w') as hf:
				hf.create_dataset('G',    data=max_n_components)
				hf.create_dataset('pros', data=gmm.weights_)
				hf.create_dataset('means',data=gmm.means_)
				hf.create_dataset('covs', data=gmm.covariances_)
				hf.create_dataset('dets', data=gmm.determinants_)
				hf.create_dataset('aic',  data=gmm.aic)
				hf.create_dataset('bic',  data=gmm.bic)
				hf.create_dataset('nmn',  data=N*np.min(gmm.weights_))
			#------------------------------------------------

			#-------------- Set as initial parameters for later smaller-component fits ---------
			init_params["weights"] = gmm.weights_
			init_params["means"]   = gmm.means_
			init_params["covs"]    = gmm.covariances_
			init_params["dets"]    = gmm.determinants_
			#--------------------------------------------------

			print("------------------------------------------")
			#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

		#----------- Sort components by determinant (useful for deterministic seeding) --------------------------
		idx = np.argsort(init_params["dets"])[::-1]
		init_params["weights"] = init_params["weights"][idx]
		init_params["means"]   = init_params["means"][idx]
		init_params["covs"]    = init_params["covs"][idx]
		init_params["dets"]    = init_params["dets"][idx]
		#-----------------------------------------------------   
		#=====================================================================================

		#-------------------- Loop over requested models and fit each -----------------------
		for n_components in np.sort(self.nc_case[case])[::-1]:
			file_model = self.file_model_base.format(instance,case,n_components)

			# Skip if model has already been saved.
			if os.path.isfile(file_model):	
				continue

			#------------- Select init_params according to components ----------
			tmp_init = {}
			tmp_init["weights"] = init_params["weights"][:n_components]
			tmp_init["means"] = init_params["means"][:n_components]
			tmp_init["covs"] = init_params["covs"][:n_components]
			tmp_init["dets"] = init_params["dets"][:n_components]
			# normalize weights to sum to 1
			tmp_init["weights"] /= np.sum(tmp_init["weights"])
			#-------------------------------------------------------------------

			try:
				#------------ Inference for the current number of components -------------
				print("Inferring model with {0} components.".format(n_components))
				gmm = self.GMM(dimension=6,n_components=n_components)
				gmm.setup(X,uncertainty=U)
				gmm.fit(tol=tolerance,
					init_min_det=init_min_det,
					init_params=tmp_init,
					random_state=self.random_state)
				#--------------------------------------------------------------------

				#------- Write the fitted model ---------------------------------------------
				with h5py.File(file_model,'w') as hf:
					hf.create_dataset('G',    data=n_components)
					hf.create_dataset('pros', data=gmm.weights_)
					hf.create_dataset('means',data=gmm.means_)
					hf.create_dataset('covs', data=gmm.covariances_)
					hf.create_dataset('dets', data=gmm.determinants_)
					hf.create_dataset('aic',  data=gmm.aic)
					hf.create_dataset('bic',  data=gmm.bic)
					hf.create_dataset('nmn',  data=N*np.min(gmm.weights_))
				#------------------------------------------------------------
			except Exception as e:
				# On failure do not stop the loop — print the error and continue.
				print(e)
				continue
			print("------------------------------------------")

		# Free memory used by X and U (they can be large)
		del X,U

	def select_best_model(self,case="Field",instance="Real",
							minimum_nmin=100,criterion="AIC"):
		"""
		Read saved model statistics (AIC/BIC) for the range of component numbers,
		select the best according to the provided criterion, and plot comparison.
		"""
		file_comparison = self.file_comparison.format(instance,case)

		#-------------------- Arrays to hold criteria per number of components----
		ngcs = np.array(self.nc_case[case],dtype=np.int16)
		aics = np.full_like(ngcs,np.nan,dtype=np.float32)
		bics = np.full_like(ngcs,np.nan,dtype=np.float32)
		nmin = np.full_like(ngcs,np.nan,dtype=np.float32)
		#-------------------------------------------------

		#--------------- Read models and their metrics ---------------------------
		for i,n_components in enumerate(self.nc_case[case]):
			file_model = self.file_model_base.format(instance,case,n_components)

			if os.path.isfile(file_model):
				#--------- Read metrics -------------------------------------------
				with h5py.File(file_model, 'r') as hf:
					assert n_components == np.array(hf.get('G')),"Error in components!"
					aics[i] = np.array(hf.get('aic'))
					bics[i] = np.array(hf.get('bic'))
					nmin[i] = np.array(hf.get('nmn'))
				#-----------------------------------------------------------

		#------ Find best model among those meeting minimum number in lightest component ----
		idx_valid = np.where(nmin > minimum_nmin)[0]
		assert len(idx_valid)>0,"Error: All {0} GMM models have nmin < {1}".format(case,minimum_nmin)
		if criterion == "BIC":
			idx_min   = np.argmin(bics[idx_valid])
		elif criterion == "AIC":
			idx_min   = np.argmin(aics[idx_valid])
		else:
			sys.exit("Criterion {0} not valid".format(criterion))
		idx_best  = idx_valid[idx_min]
		#--------------------------------------------------

		#------------ Register the best model chosen for this instance ------------
		if instance in self.best_gmm.keys():
			self.best_gmm[instance].update({case:int(ngcs[idx_best])})
		else:
			self.best_gmm.update({instance:{case:int(ngcs[idx_best])}})
		#-----------------------------------------------------------------------

		#-----------Plot BIC,AIC,NMIN ------------------------
		plt.figure(figsize=(8,6))
		axl = plt.gca()
		axl.plot(ngcs,bics,label="BIC")
		axl.plot(ngcs,aics,label="AIC")
		axl.set_xlabel('Components')
		axl.set_ylabel("Criterion")
		axl.set_xticks(ngcs)
		axl.legend(loc="upper left")

		axr = axl.twinx()
		axr.plot(ngcs,nmin,
				ls="--",color="black",label="$N_{min}$")
		axr.set_yscale("log")
		axr.set_ylabel("N_stars in lightest components")
		axr.legend(loc="upper right")

		axr.axvline(x=ngcs[idx_best],
					linewidth=1, color='gray',ls=":")

		plt.savefig(file_comparison,bbox_inches='tight')
		plt.close()
		#-----------------------------------------------------

	def plot_model(self,n_components=None,case="Field",instance="Real"):
		"""
		Visualize the fitted GMM model and the data in 2x2 panels for positions and velocities.
		A PDF file with subplots per pair is written next to the model HDF5 file.
		"""
		#--------------- Determine n_components ----------------------------------------
		if n_components is None:
			n_components = self.best_gmm[instance][case]
		assert isinstance(n_components,int), "n_components must be an integer!"
		#----------------------------------------------------------------------

		file_data  = self.file_data_base.format(instance)
		file_model = self.file_model_base.format(instance,case,n_components)

		#---------- Avoid recomputing if PDF already present ----------
		if os.path.isfile(file_model.replace(".h5",".pdf")):
			return
		#----------------------------------------------------

		#--------- Read model params -----------------------------
		with h5py.File(file_model, 'r') as hf:
			n_components = np.array(hf.get('G'))
			weights      = np.array(hf.get('pros'))
			means        = np.array(hf.get('means'))
			covariances  = np.array(hf.get('covs'))
		#--------------------------------------------------

		#------------ Read data --------------------
		with h5py.File(file_data, 'r') as hf:
			X = np.array(hf.get("mu_"+case))
		#--------------------------------------

		#--------------------------------------------------------
		pdf = PdfPages(filename=file_model.replace(".h5",".pdf"))

		#------------ Colorbar for features ----------------------------
		norm_feat = Normalize(vmin=min(weights),vmax=max(weights))

		# legend handles
		handles = [mlines.Line2D([], [], color="black", marker=None,
					linestyle="-",label="Model"),
				   mlines.Line2D([], [], color="w", marker=".",
					markerfacecolor="black",markersize=5,label="Data") ]

		sm = cm.ScalarMappable(norm=norm_feat, cmap=self.cmap_feat)
		#---------------------------------------------------------

		#------ Positions  -----------------------------------------
		fig, axs = plt.subplots(nrows=2,ncols=2,figsize=None)
		for ax,idx in zip([axs[0,0],axs[0,1],axs[1,0]],self.idxs[:3]):
			mxgr = np.ix_(range(n_components),idx,idx)

			#--------- Plot data points --------------------------
			ax.scatter(x=X[:,idx[0]],y=X[:,idx[1]],
						marker=".",s=1,
						c="gray",zorder=0,
						rasterized=True,
						label="Data")
			#-------------------------------------------

			#---------------- Overlay inferred ellipses for each GMM component ------------------------------
			for w,mu,sg in zip(weights,means[:,idx],covariances[mxgr]):
				width, height, angle = get_principal(sg)
				ell  = Ellipse(mu,width=width,height=height,angle=angle,
						clip_box=ax.bbox,edgecolor=sm.cmap(sm.norm(w)),
						ls="-",fill=False)
				ax.add_patch(ell)

			# Plot component centers
			ax.scatter(means[:,idx[0]],means[:,idx[1]],
						c=sm.cmap(sm.norm(weights)),marker='s',s=1)
			#--------------------------------------------------------------

			#------------- Titles / axis labels --------------------
			ax.set_xlabel(self.OBS[idx[0]])
			ax.set_ylabel(self.OBS[idx[1]])
			ax.locator_params(tight=True, nbins=3)
			#----------------------------------------

		# Hide some axis labels for compact layout, add colorbar and legend
		axs[0,0].axes.xaxis.set_visible(False)
		axs[0,1].axes.yaxis.set_visible(False)
		axs[1,1].axis("off")
		axs[1,1].legend(handles=handles,loc='center',
						bbox_to_anchor=(0.25, 0., 0.5, 0.5))
		fig.colorbar(sm,ax=axs[1,1],fraction=0.3,anchor=(0.0,0.0),
			shrink=0.75,extend="both",label='Weights')
		plt.subplots_adjust(left=None, bottom=None, right=None,
						top=None, wspace=0.0, hspace=0.0)
		pdf.savefig(dpi=100,bbox_inches='tight')
		plt.close()

		#------ Velocities -----------------------------------------
		fig, axs = plt.subplots(nrows=2,ncols=2,figsize=None)

		for ax,idx in zip([axs[0,0],axs[0,1],axs[1,0]],self.idxs[3:]):
			mxgr = np.ix_(range(n_components),idx,idx)

			#--------- Plot data points --------------------------
			ax.scatter(x=X[:,idx[0]],y=X[:,idx[1]],
						marker=".",s=1,
						c="gray",zorder=0,
						rasterized=True,
						label="Data")
			#-------------------------------------------

			#---------------- Overlay ellipses -----------------------------------
			for w,mu,sg in zip(weights,means[:,idx],covariances[mxgr]):
				width, height, angle = get_principal(sg)
				ell  = Ellipse(mu,width=width,height=height,angle=angle,
						clip_box=ax.bbox,edgecolor=sm.cmap(sm.norm(w)),
						ls="-",fill=False)
				ax.add_patch(ell)

			ax.scatter(means[:,idx[0]],means[:,idx[1]],
						c=sm.cmap(sm.norm(weights)),marker='s',s=1)
			#--------------------------------------------------------------

			#------------- Titles --------------------
			ax.set_xlabel(self.OBS[idx[0]])
			ax.set_ylabel(self.OBS[idx[1]])
			ax.locator_params(tight=True, nbins=3)
			#----------------------------------------

		axs[0,0].axes.xaxis.set_visible(False)
		axs[0,1].axes.yaxis.set_visible(False)
		axs[1,1].axis("off")
		axs[1,1].legend(handles=handles,loc='center',
						bbox_to_anchor=(0.25, 0., 0.5, 0.5))
		fig.colorbar(sm,ax=axs[1,1],fraction=0.3,anchor=(0.0,0.0),
			shrink=0.75,extend="both",label='Weights')
		plt.subplots_adjust(left=None, bottom=None, right=None,
						top=None, wspace=0.0, hspace=0.0)
		pdf.savefig(dpi=100,bbox_inches='tight')
		plt.close()
		pdf.close()

		del X

	def compute_probabilities(self,instance="Real",
					chunks=1,replace=False,use_prior=False):
		"""
		Compute membership probabilities for all sources in the dataset:
		- Load best GMM models for Field and Cluster (self.best_gmm must be set).
		- Compute log-likelihoods for each source under each model, taking
		  into account per-source covariance (uncertainty).
		- Optionally include prior ratio based on sample sizes.
		- Save computed probabilities to the HDF5 file.
		"""
		file_data  = self.file_data_base.format(instance)

		#------- Read data----------------------------------
		print("Reading data ...")
		with h5py.File(file_data, 'r') as hf:
			# If probabilities already exist and replace is False, ensure consistency and exit early.
			if self.PRO in hf.keys() and not replace:
				cls_ngmm = np.array(hf.get("Cluster_nGMM"))
				fld_ngmm = np.array(hf.get("Field_nGMM"))
				assert cls_ngmm == self.best_gmm[instance]["Cluster"],\
				"ERROR: different best cluster model from that used for probabilities"
				assert fld_ngmm == self.best_gmm[instance]["Field"],\
				"ERROR: different best field model from that used for probabilities"
				return
			mu = np.array(hf.get("mu"))
			sg = np.array(hf.get("sg"))
			idx_cls = np.array(hf.get("idx_Cluster"))
		#---------------------------------------------------

		#-- Dimensions --
		N,_ = mu.shape
		nc  = len(idx_cls)
		nf  = N - nc
		#----------------

		# Optionally include a prior ratio between field and cluster sizes in log odds.
		if use_prior:
			ln_prior_ratio = np.log(np.float64(nf)/np.float64(nc))
		else:
			ln_prior_ratio = 0.0

		#------- Chunking: partition the array of size N into 'chunks' blocks
		proc_n = [ N // chunks + (N % chunks > n)
						for n in range(chunks)]
		pos = 0
		pos_n = []
		for n in range(chunks):
			pos_n.append(pos)
			pos += proc_n[n]

		chunk_n    = [proc_n[rank] for rank in range(chunks)]
		chunk_off  = [pos_n[rank] for rank in range(chunks)]
		chunk_idx  = [range(chunk_off[c],chunk_off[c]+chunk_n[c])
						for c in range(chunks)]

		#----- Likelihoods placeholder (N x 2) for Field and Cluster
		llks = np.zeros((N,2))
		for i,case in enumerate(["Field","Cluster"]):
			n_components = self.best_gmm[instance][case]
			#--------------- Read model file for this case --------------------------
			file_model = self.file_model_base.format(
				instance,case,n_components)
			#------------------------------------------------

			#------------ Read model parameters -------------------
			print("Reading {0} parameters ...".format(case))
			with h5py.File(file_model, 'r') as hf:
				n_components = np.array(hf.get("G"))
				weights      = np.array(hf.get("pros"))
				means        = np.array(hf.get("means"))
				covariances  = np.array(hf.get("covs"))
			#-------------------------------------------

			#------------ Compute log-likelihoods in chunks ------------------------------------
			print("Computing likelihoods ...")
			gmm = self.GMM(dimension=6,n_components=n_components)

			for idx in chunk_idx:
				# Setup GMM with mu and U for the subset and compute log-likelihoods
				gmm.setup(mu[idx],uncertainty=sg[idx])
				# logsumexp applied to mixture log-likelihoods provides log p(x).
				llks[idx,i] = logsumexp(gmm.log_likelihoods(
										weights,means,covariances),
										axis=1,keepdims=False)
			del gmm
			#-----------------------------------------------------------

		# Safety checks: ensure computed likelihoods are finite
		assert np.all(np.isfinite(llks)), "Likelihoods are not finite!"
		del mu,sg
		#--------------------------------------------------------------

		#------- Convert log-likelihoods into membership probabilities
		print("Computing probabilities ...")
		pc = 1.0/(1.0+np.exp(ln_prior_ratio + llks[:,0] - llks[:,1]))
		del llks
		assert np.all(np.isfinite(pc)), "Probabilities are not finite!"
		assert np.all(pc >= 0.0), "Probabilities are negative!"
		assert np.all(pc <= 1.0), "Probabilities are larger than one!"
		#--------------------------------------------------------------------

		#----------- Print cluster probabilities summary for known members
		print("Cluster probabilities:")
		print("Min:{0:1.4f}, Mean:{1:1.4f}, Median:{2:1.4f}, Max:{3:1.4f}".format(
			np.min(pc[idx_cls]),np.mean(pc[idx_cls]),
			np.median(pc[idx_cls]),np.max(pc[idx_cls])))
		#------------------------------------------------------------------

		#---------- Save probability array to HDF5 -------------------------
		print("Saving probabilities ...")
		with h5py.File(file_data, 'a') as hf:
			if replace:
				# Remove existing datasets if requested
				del hf[self.PRO]
				del hf["Cluster_nGMM"]
				del hf["Field_nGMM"]
			hf.create_dataset(self.PRO,data=pc)
			hf.create_dataset("Cluster_nGMM",data=self.best_gmm[instance]["Cluster"])
			hf.create_dataset("Field_nGMM",data=self.best_gmm[instance]["Field"])
		#-----------------------------------------------------------

	def generate_synthetic(self,n_cluster=int(1e5),seeds=range(1)):
		"""
		Call Amasijo.generate_cluster for several seeds and create per-seed
		directories with synthetic members CSVs. This method writes the
		member CSVs, leaving HDF5 assembly to assemble_synthetic.
		"""
		#================= Generate synthetic data ====================
		for seed in seeds:
			#--------- Directory ---------------------------------
			name_base = "/Synthetic_{0}/".format(seed)
			dir_sim   = self.dir_main + name_base
			file_smp  = self.file_smp_base.format(name_base)
			#----------------------------------------------------

			if os.path.isfile(file_smp):
			   continue 

			print(" Generating cluster data of seed {0} ...".format(seed))

			#--- Create simulation directory -----------
			os.makedirs(dir_sim,exist_ok=True)
			os.makedirs(dir_sim + "Data",exist_ok=True)
			os.makedirs(dir_sim + "Models",exist_ok=True)
			#-------------------------------------------

			#---------- Generate cluster ---------------------------
			ama = Amasijo(isochrones_args=self.isochrones_args,
						kalkayotl_args={
						"file":self.file_par_kal,
						"statistic":self.kalkayotl_args["statistic"],
						"velocity_model":"joint"},
						photometry={
						"labels":{
							"phot_g_mean_mag":"phot_g_mean_mag",
							"phot_bp_mean_mag":"phot_bp_mean_mag",
							"phot_rp_mean_mag":"phot_rp_mean_mag"},
						"family":"Gaia"},
						radial_velocity={
						"labels":{"radial_velocity":"radial_velocity"},
						"family":"Gaia"},
						seed=seed)
			# Create CSV with synthetic members
			ama.generate_cluster(file_smp,n_stars=n_cluster,
				angular_correlations=None)
			# ama.plot_cluster(file_plot=file_smp.replace(".csv",".pdf"))
			del ama
			#----------------------------------------------------------

	def assemble_synthetic(self,seeds=[0]):
		"""
		For each synthetic seed:
		- Read generated members CSV and create per-seed HDF5 files with mu_Cluster, sg_Cluster, etc.
		- Use real-field data extracted from the 'Real' dataset to assemble combined datasets
		  (cluster + field) saved for each synthetic seed instance.
		"""
		# Choose columns from self.observables excluding correlation fields and ruwe
		columns    = [ obs for obs in self.observables \
						if (obs not in self.RHO) and (obs != "ruwe")]
		local_seeds = seeds.copy()

		#-------------- Check if data exists for seeds and skip those already assembled -----
		for seed in seeds:
			#--------- Directory ---------------------------------
			name_base = "/Synthetic_{0}/".format(seed)
			dir_sim   = self.dir_main + name_base
			file_data = self.file_data_base.format(name_base)
			#----------------------------------------------------

			if os.path.isfile(file_data):
				with h5py.File(file_data, 'r') as hf:
					if "mu" in hf.keys():
						local_seeds.remove(seed)
		#---------------------------------------------------------

		#-------------------------
		if len(local_seeds) == 0:
			return
		#-------------------------

		#================= Cluster ==================================
		for seed in local_seeds:
			#--------- Directory ---------------------------------
			name_base = "/Synthetic_{0}/".format(seed)
			dir_sim   = self.dir_main + name_base
			file_smp  = self.file_smp_base.format(name_base)
			file_data = self.file_data_base.format(name_base)
			#----------------------------------------------------

			if os.path.isfile(file_data):
			   continue

			print("Saving cluster data of seed {0} ...".format(seed))

			#-------- Read cluster CSV and compute uncertainties placeholders -------
			df_cls = pd.read_csv(file_smp,usecols=columns)
			# Set ruwe to unity for synthetic stars as default
			df_cls["ruwe"] = 1.0
			n_cls  = len(df_cls)
			#---------------------------------------------

			#----- Select members indices -----------------
			idx_cls  = np.arange(n_cls)
			#------------------------------------------------

			#------ Extract arrays for storage ----------
			mu_cls = df_cls.loc[:,self.OBS].to_numpy()
			sd_cls = df_cls.loc[:,self.UNC].to_numpy()
			ex_cls = df_cls.loc[:,self.EXT].to_numpy()
			del df_cls
			#-----------------------------------------

			#----------- Covariance matrices creation for synthetic cluster (zeros)
			zeros = np.zeros((len(sd_cls),6,6))
			diag = np.einsum('...jj->...j',zeros)
			# diag filled with squared sd (variance) on the diagonal
			diag[:] = np.square(sd_cls)
			sg_cls = zeros.copy()
			del diag,zeros,sd_cls
			#----------------------------------------

			#--------------- Write per-seed HDF5 cluster data ------------------------------------
			with h5py.File(file_data, 'w') as hf:
				hf.create_dataset('mu_Cluster', data=mu_cls)
				hf.create_dataset('sg_Cluster', data=sg_cls)
				hf.create_dataset('ex_Cluster', data=ex_cls)
				hf.create_dataset('idx_Cluster',data=idx_cls)
			#----------------------------------------------------------------
			del idx_cls,mu_cls,sg_cls,ex_cls

		#=============== Field ================================
		file_data  = self.file_data_base.format("Real")
		print("Loading real data ...")

		#------ Read real probabilities and extract field subset ---------------
		with h5py.File(file_data, 'r') as hf:
			mu = np.array(hf.get("mu"))
			sg = np.array(hf.get("sg"))
			ex = np.array(hf.get("ex"))
			idx_field = np.array(hf.get("idx_Field"))
		#------------------------------------------

		print("Selecting field ...")
		mu_fld  = mu[idx_field]
		sg_fld  = sg[idx_field]
		ex_fld  = ex[idx_field]
		n_field = len(idx_field)

		del mu,sg,ex,idx_field

		#================= Cluster + Field combination for each seed ==================================
		for seed in local_seeds:
			print("Saving data of seed {0} ...".format(seed))
			#--------- Directory ---------------------------------
			name_base = "/Synthetic_{0}/".format(seed)
			dir_sim   = self.dir_main + name_base
			file_data = self.file_data_base.format(name_base)
			#----------------------------------------------------

			#-------- Read cluster data -------------------
			with h5py.File(file_data, 'r') as hf:
				idx_cls = np.array(hf.get("idx_Cluster"))
				mu_cls  = np.array(hf.get("mu_Cluster"))
				sg_cls  = np.array(hf.get("sg_Cluster"))
				ex_cls  = np.array(hf.get("ex_Cluster"))
			#-------------------------------------------------

			#----- Assign field indices that follow cluster indices in concatenated array
			idx_fld  = np.arange(len(idx_cls),len(idx_cls)+n_field)
			#------------------------------------------------------

			#------- Concatenate cluster and field arrays to produce full dataset per-seed -----------
			mu_data = np.concatenate((mu_cls,mu_fld),axis=0)
			sg_data = np.concatenate((sg_cls,sg_fld),axis=0)
			ex_data = np.concatenate((ex_cls,ex_fld),axis=0)
			#-----------------------------------------------

			#--------------- Write concatenated data into per-seed HDF5 file------------------------------------
			with h5py.File(file_data, 'a') as hf:
				hf.create_dataset('mu',         data=mu_data)
				hf.create_dataset('sg',         data=sg_data)
				hf.create_dataset('ex',         data=ex_data)
				hf.create_dataset('idx_Field',  data=idx_fld)
			#----------------------------------------------------------------
			del idx_fld,mu_data,sg_data,ex_data

	def compute_probabilities_synthetic(self,seeds,chunks=1,
							replace=False,use_prior=False):
		"""
		Compute probabilities for previously assembled synthetic datasets by:
		- copying the model files from 'Real' instance to the synthetic instance
		- running compute_probabilities using the copied models
		"""
		for i,seed in enumerate(seeds):
			#------------ File and directory for synthetic instance ----------------------------
			instance  = "Synthetic_{0}".format(seed)
			dir_model = "{0}/{1}/Models/".format(self.dir_main,instance)
			os.makedirs(dir_model,exist_ok=True)
			#-------------------------------------------------------------

			#-------- Same models as real data: just register them in best_gmm dict ---------------
			self.best_gmm[instance] = self.best_gmm["Real"]
			#-------------------------------------------------------

			#---------------- Copy best models from Real to synthetic model directory ------------------------
			model_fld = self.file_model_base.format("Real","Field",
						self.best_gmm["Real"]["Field"])
			model_cls = self.file_model_base.format("Real","Cluster",
						self.best_gmm["Real"]["Cluster"])

			cmd_fld   = "cp {0} {1}".format(model_fld,dir_model)
			cmd_cls   = "cp {0} {1}".format(model_cls,dir_model)

			os.system(cmd_fld)
			os.system(cmd_cls)
			#-----------------------------------------------------

			#------------------------------------------------------------
			print(30*"-")
			print("Computing probabilities of seed {0} ...".format(seed))
			self.compute_probabilities(instance=instance,
									chunks=chunks,
									replace=replace,
									use_prior=use_prior)
			print(30*"-")
			#------------------------------------------------------------

	def find_probability_threshold(self,seeds,bins=4,
		covariate="phot_g_mean_mag",metric="MCC",covariate_limits=None,
		plot_log_scale=False,
		prob_steps={
				0.954499736103642:10, # 2sigma
				0.997300203936740:10, # 3sigma
				0.999936657516334:10, # 4sigma
				0.999999426696856:10 # 5sigma
				},
		min_prob=0.682689492137086):
		"""
		Using synthetic experiments (seeds), evaluate classifier performance
		for different probability thresholds and covariate bins using the
		ClassiferQuality helper from Amasijo. Saves threshold data (dill),
		plots and TeX summary files.
		"""
		file_plot = self.file_qlt_base.format(covariate,metric,"pdf")
		file_tex  = self.file_qlt_base.format(covariate,metric,"tex")
		file_thr  = self.file_qlt_base.format(covariate,metric,"pkl")

		os.makedirs(self.dir_main+"/Classification/",exist_ok=True)

		# Collect DataFrame per seed with true class (Cluster or not), prob and covariates
		dfs = []
		for seed in seeds:
			print("Reading seed {0}".format(seed))
			#------------ File ----------------------------
			instance  = "Synthetic_{0}".format(seed)
			file_data = self.file_data_base.format(instance)

			#------ Read data and probabilities -----------------------
			with h5py.File(file_data, 'r') as hf:
				idx_cls  = np.array(hf.get("idx_Cluster"))
				idx_fld  = np.array(hf.get("idx_Field"))
				pc = np.array(hf.get(self.PRO))
				ex_data  = np.array(hf.get("ex"))
			#-------------------------------------------------

			#---------- Class: boolean mask for true cluster membership ----
			classs = np.full(len(pc),fill_value=False)
			classs[idx_cls] = True
			#-----------------------------------------

			#------ Create dataframe with labels and probabilities ----------------
			df = pd.DataFrame(data={"Cluster":classs,
									self.PRO:pc})
			#----------------------------------------

			#-------- Insert covariates into dataframe -------------------------
			for ex,name in zip(ex_data.T,self.EXT):
				df.insert(loc=2,column=name,value=ex)
			#----------------------------------------

			#-- Append the per-seed df to list ----
			dfs.append(df)
			#--------------

			del df

		print("Analyzing classifier quality ...")
		# ClassifierQuality uses list of DataFrames, computes confusion matrices, thresholds, etc.
		clq = ClassifierQuality(file_data=dfs,
								variate=self.PRO,
								covariate=covariate,
								covariate_limits=covariate_limits,
								true_class="Cluster")
		del dfs

		print("Computing confusion matrices ...")
		clq.confusion_matrix(bins=bins,
							prob_steps=prob_steps,
							metric=metric,
							min_prob=min_prob)

		print("Plotting and saving quality measures ...")
		clq.plots(file_plot=file_plot,log_scale=plot_log_scale)
		clq.save(file_tex=file_tex)

		# Save thresholds object path for later use by select_members
		self.file_thresholds = file_thr

		del clq

	def select_members(self,probability_threshold=None,instance="Real"):
		"""
		Select candidates based on a global probability threshold (float) or
		using a thresholds object generated by find_probability_threshold. The
		method:
		- loads HDF5 arrays (ids, mu, sd, cr, ex, pc)
		- builds a DataFrame with observables and uncertainties
		- determines candidate members according to thresholds or fixed probability
		- saves the candidate CSV and writes a multi-page PDF with diagnostic plots
		"""
		file_data = self.file_data_base.format(instance)

		#------ Read probabilities and supporting arrays --------------------------
		with h5py.File(file_data, 'r') as hf:
			idx_cls = np.array(hf.get("idx_Cluster"))
			ids = np.array(hf.get("ids"),dtype=np.uint64)
			mu = np.array(hf.get("mu"))
			sd = np.array(hf.get("sd"))
			cr = np.array(hf.get("cr"))
			ex = np.array(hf.get("ex"))
			pc = np.array(hf.get(self.PRO))
		#----------------------------------------------------

		#-------- Build a pandas dataframe by concatenating arrays ----------------------
		names = sum([self.OBS,self.UNC,self.RHO,self.EXT],[])
		dt = np.hstack((mu,sd,cr,ex))
		df_cat = pd.DataFrame(data=dt,columns=names)
		df_cat.insert(loc=0,column=self.PRO,value=pc)
		df_cat.insert(loc=0,column=self.IDS,value=ids)
		#---------------------------------------------------------

		#----- Members subset (the "true" members provided earlier) ---------------
		df_mem  = df_cat.iloc[idx_cls]
		#-----------------------------

		#----- Candidate selection: either a single threshold or thresholds per bin ----
		print("Selecting candidates ...")
		if isinstance(probability_threshold,float):
			idx_cnd = np.where(pc >= probability_threshold)[0]
			df_cnd  = df_cat.iloc[idx_cnd]

		else:
			#------------- Load thresholds object (dill) -----------------------------------
			file_thresholds = self.file_thresholds if probability_threshold \
						is None else probability_threshold
			#----------------------------------------------------------------

			#----- Load edges and probability thresholds ------
			with open(file_thresholds,'rb') as in_stream:
				quality = dill.load(in_stream)
			#--------------------------------------------------

			#------- Split data frame into bins according to covariate (e.g., magnitude) -----
			bin_mag = np.digitize(df_cat[quality["covariate"]].values,
						bins=quality["edges"])
			#--------------------------------------------------------

			#------ Bin 0 objects to bin 1 (protect against objects brighter than brightest edge)
			bin_mag[np.where(bin_mag == 0)[0]] = 1
			#-------------------------------------

			#----------- Loop over bins and apply per-bin thresholds ----------------------------
			dfs = []
			for i,threshold in enumerate(quality["thresholds"]):
				#--------- Objects in bin -----------------------
				idx = np.where(bin_mag == i)[0]
				strategy = "Bin {0}".format(i)
				# There are no objects in bin zero so we use it for all objects
				if i == 0:
					# In this design, bin 0 is reserved and should be empty — if not, throw.
					assert len(idx) == 0 ,"Bin 0 is not empty!"
					idx = np.arange(len(df_cat))
					strategy = "All"
				#------------------------------------------------

				#------------- Members in this bin that pass threshold ---------------------
				idx_mem = np.where(pc[idx] >= threshold)[0]
				tmp_mem = df_cat.iloc[idx[idx_mem]].copy()
				tmp_mem["Strategy"] = strategy
				#-------------------------------------------

				#------------ Append the bin's selected members ---------------------------
				dfs.append(tmp_mem)
				#---------------------------------------------------------------

			#--------- Concatenate, exclude the "All" strategy and keep per-bin results -----
			df = pd.concat(dfs)
			df_cnd = df[df["Strategy"] != "All"].copy()
			del df_cat,dfs,df
			#-------------------------------------------
		#-------------------------------------------------

		#---------- IDs and summary ------------------------
		ids_cnd = df_cnd[self.IDS].to_numpy()
		ids_mem = df_mem[self.IDS].to_numpy()
		#---------------------------------------

		#-------- Summary of overlap between provided members and selected candidates ----------
		ids_common = np.intersect1d(ids_mem,ids_cnd)
		ids_reject = np.setdiff1d(ids_mem,ids_cnd)
		ids_new    = np.setdiff1d(ids_cnd,ids_mem)
		print("Common: {0}".format(len(ids_common)))
		print("Rejected: {0}".format(len(ids_reject)))
		print("New: {0}".format(len(ids_new)))
		#---------------------------------------------

		#--------- Save candidates as CSV ------------------
		df_cnd.to_csv(self.file_mem_data,index=False)
		#--------------------------------------------

		#----------- Prepare colors/columns for plotting (compute g-rp color if not present) -----------
		if "g_rp" not in df_mem.columns:
			df_mem["g_rp"] = df_mem["phot_g_mean_mag"] - \
							 df_mem["phot_rp_mean_mag"]
		if "g_rp" not in df_cnd.columns:
			df_cnd["g_rp"] = df_cnd["phot_g_mean_mag"] - \
							 df_cnd["phot_rp_mean_mag"]
		#---------------------------------------------

		#----------- Compute absolute magnitude G using parallax ------------------
		df_mem["G"] = df_mem["phot_g_mean_mag"] + 5.*( 1.0 -
						np.log10(1000./df_mem["parallax"]))
		df_cnd["G"] = df_cnd["phot_g_mean_mag"] + 5.*( 1.0 -
						np.log10(1000./df_cnd["parallax"]))
		#--------------------------------------------------

		#--------------------------------------------------------
		# Create a multi-page PDF with plots comparing current members and candidates
		pdf = PdfPages(filename=self.file_mem_plot)
		for i,plot in enumerate(self.plots):
			fig = plt.figure()
			ax  = plt.gca()
			#--------- Members plotted as colored points (by probability) --------------------------
			ax.scatter(x=df_mem[plot[0]],y=df_mem[plot[1]],
						marker=".",s=5,
						c=df_mem[self.PRO],
						vmin=0,vmax=1,
						cmap=self.cmap_prob,
						zorder=0,
						label="Members")
			scb = ax.scatter(x=df_cnd[plot[0]],y=df_cnd[plot[1]],
						marker="$\u25A1$",
						s=10,c=df_cnd[self.PRO],
						vmin=0,vmax=1,
						cmap=self.cmap_prob,
						zorder=0,
						label="Candidates")
			#-------------------------------------------

			#------------- Titles --------------------
			ax.set_xlabel(plot[0])
			ax.set_ylabel(plot[1])
			#-----------------------------------------
			ax.locator_params(tight=True, nbins=10)
			#----------------------------------------

			#------- Invert y-axis for magnitude plots ----------
			if i>=3:
				ax.invert_yaxis()
			#-----------------------

			ax.legend()
			fig.colorbar(scb,shrink=0.75,extend="both",label='Probability')
			pdf.savefig(dpi=100,bbox_inches='tight')
			plt.close()
		pdf.close()
		#-------------------------------------------------------------------
		#===================================================================

	def run_real(self,file_catalogue,file_members,
				n_cluster=int(1e5),n_field=int(1e5),
				replace_probabilities=False,
				use_prior_probabilities=False,
				best_model_criterion="AIC",
				init_min_det=1e-3,
				minimum_nmin=100,
				chunks=1):
		"""
		Run the full pipeline for the real dataset:
		- generate synthetic cluster if missing
		- assemble data HDF5 (catalogue + members + synthetic)
		- infer GMM models for Field and Cluster
		- select best GMM (AIC/BIC)
		- plot the models
		- compute probabilities for the real dataset
		"""
		#-------------------- Synthetic --------------------------
		if not os.path.isfile(self.file_smp_base.format("Real")):
			self.generate_true_cluster(n_cluster=n_cluster,
					file_kalkayotl=self.file_par_kal)
		#---------------------------------------------------------

		#------------- Assemble real HDF5 dataset if missing -----------
		if not os.path.isfile(self.file_data_base.format("Real")):
			self.assemble_data(file_catalogue=file_catalogue,
								file_members=file_members,
								n_field=n_field,
								instance="Real")
		#------------------------------------------------------

		#--------------- Infer models for Field and Cluster ---------------------
		self.infer_models(case="Field",  instance="Real",
				init_min_det=init_min_det)
		self.infer_models(case="Cluster",instance="Real",
				init_min_det=init_min_det)
		#-------------------------------------------------

		#------------- Select best models (if not already chosen) --------------------------
		if "Real" not in self.best_gmm:
			self.select_best_model(case="Field",instance="Real",
									minimum_nmin=minimum_nmin,
									criterion=best_model_criterion)
			self.select_best_model(case="Cluster",instance="Real",
									minimum_nmin=minimum_nmin,
									criterion=best_model_criterion)
			print("The best real GMM models are:")
			print(self.best_gmm["Real"])
		#-----------------------------------------------------------

		#----------------- Plot chosen best models ---------------
		self.plot_model(case="Field",instance="Real")
		self.plot_model(case="Cluster",instance="Real")
		#--------------------------------------------------

		#-------- Probabilities for the real dataset --------------------------------
		self.compute_probabilities(instance="Real",
							chunks=chunks,
							replace=replace_probabilities,
							use_prior=use_prior_probabilities)
		#--------------------------------------------------------

	def run_synthetic(self,seeds,
					n_cluster=int(1e5),chunks=1,
					replace_probabilities=False,
					use_prior_probabilities=False):
		"""
		Generate/assemble synthetic datasets and compute probabilities for them.
		"""
		#----------- Synthetic data --------------------------------
		self.generate_synthetic(n_cluster=n_cluster,
							   seeds=seeds)
		self.assemble_synthetic(seeds=seeds)
		self.compute_probabilities_synthetic(seeds,
						chunks=chunks,
						replace=replace_probabilities,
						use_prior=use_prior_probabilities)
		#----------------------------------------------------------

	def filter_members(self,file_input,file_output,args):
		"""
		Filter the input members list according to configured criteria:
		- minimum probability (prob_cls)
		- magnitude limit
		- radial velocity uncertainty limits
		- ruwe threshold (binary candidate removal)
		- radial velocity outlier clipping
		Filtered list is saved to file_output.
		"""
		#============= Load members =========================
		#----- Load catalogue ------------------------
		if '.csv' in file_input:
			df = pd.read_csv(file_input)
		elif ".fits" in file_input:
			dat = Table.read(file_input, format='fits')
			df  = dat.to_pandas()
			del dat
		else:
			sys.exit("Format file not recognized. Only CSV of FITS")
		#-------------------------------------------------------------
		#--------- Rename and verify prob_cls -----------------------------------------------
		df.rename(columns=self.input_mapper,inplace=True)
		if "prob_cls" not in df.columns:
			df["prob_cls"] = 1.0
			print("WARNING: prob_cls is not present in input members file and was set to 1.0")
		#------------------------------------------------------------------------------------
		#==============================================================

		#------------ Probability filter -------------------------
		df = df.loc[df["prob_cls"]>= args["prob_threshold"]]
		#---------------------------------------------------------

		#------- Drop faint members ---------------
		if args["g_mag_limit"] is not None:
			df = df.loc[df["phot_g_mean_mag"] < args["g_mag_limit"]]
		#----------------------------------------------

		#--- Assert that observed values have uncertainties and viceversa ----
		nan_rvs = np.isnan(df["radial_velocity"].values)
		nan_unc = np.isnan(df["radial_velocity_error"].values)
		np.testing.assert_array_equal(nan_rvs,nan_unc,
		err_msg="There are discrepant rvs missing uncertainties and values!")
		#---------------------------------------------------------------------

		print("Replacing minimum and maximum uncertainties ...")
		#----------- Set minimum uncertainty -------------------------------------
		condition = df["radial_velocity_error"] < args["rv_error_limits"][0]
		df.loc[condition,"radial_velocity_error"] = args["rv_error_limits"][0]
		#-------------------------------------------------------------------------
		#----------- Set maximum uncertainty -------------------------------------
		condition = df["radial_velocity_error"] > args["rv_error_limits"][1]
		df.loc[condition,"radial_velocity"] = np.nan
		df.loc[condition,"radial_velocity_error"]  = np.nan
		#-------------------------------------------------------------------------
		#------------- Drop Binaries based on ruwe threshold --------------------------
		n_binaries = sum(df["ruwe"] >= args["ruwe_threshold"])
		if n_binaries > 0:
			df = df.loc[df["ruwe"] < args["ruwe_threshold"]]
		print("Dropped binaries: {0}".format(n_binaries))
		#-----------------------------------------------------

		#---------- Drop outliers (radial velocity clipping) ------------------------
		mu_rv = np.nanmean(df["radial_velocity"])
		sd_rv = np.nanstd(df["radial_velocity"])
		print("Radial velocity: {0:2.1f} +/- {1:2.1f} km/s".format(mu_rv,sd_rv))
		maha_dst = np.abs(df["radial_velocity"] - mu_rv)/sd_rv
		n_rvout = sum(maha_dst >= args["rv_sd_clipping"])
		if n_rvout > 0:
			df = df.loc[maha_dst < args["rv_sd_clipping"]]
		print("Dropped RV outliers: {0}".format(n_rvout))
		#----------------------------------------------------------------------------

		#------------------------ Optionally drop records missing RV ------------------------
		if not args["allow_rv_missing"]:
			df.dropna(subset=["radial_velocity","radial_velocity_error"],
				inplace=True)
		#------------------------------------------------------------------

		print("Saving filtered members ...")
		df.to_csv(file_output)
		#==================================================================

	def clean_members(self,file_input,file_output,
		clean_args = {},
		parameters = {},
		hyper_parameters={},
		parameterization="central"):
		"""
		A multi-stage cleaning routine that runs Kalkayotl Inference in iterative
		FGMM (Finite GMM) steps. This method:
		- fills default args if missing
		- for each count (iteration) either copies or merges previous classification
		- calls Kalkayotl inference (self.Inference) and saves statistics
		- checks convergence diagnostics and stops early if field weight is negligible
		"""
		#---------------- Default arguments -------------------
		default_clean_args ={
			"input":None,
			"counts":[0,1,2,3],
			"min_field_frac":0.1,
			"conv_r_hat":1.1,
			"tuning_iters":int(1e5),
			"sample_iters":2000,
			"target_accept":0.65,
			"chains":4,
			"cores":4,
			"init_iters":int(1e6),
			"init_refine":False,
			"prior_predictive":True,
			"prior_iters":1000,
			"progressbar":True,
			"nuts_sampler":"advi",
			"random_seed":12345}

		default_parameters = {
					"location":None,
					"scale":None,
					"weights":None,
					"field_scale":[20.,20.,20.,5.,5.,5.]}

		default_hyper_parameters={
					"location":None,
					"scale":None,
					"weights":{"a":np.array([9,1])},
					"eta":None}
		#---------------------------------------------------

		#------------- Use provided arguments ---------------
		for default,given in zip(
			[default_clean_args,
			default_parameters,
			default_hyper_parameters],
			[clean_args,parameters,hyper_parameters]):
			for arg,val in default.items():
				if not arg in given:
					given[arg] = val
			print("The following arguments will be used:")
			for k,v in given.items():
				print("{0} : {1}".format(k,v))
		#--------------------------------------------------

		#--------------- Files -----------------------------
		dir_base  = self.dir_main +"/Kalkayotl/"
		file_mem = dir_base + "{0}/members.csv"
		file_src = dir_base + "{0}/Sources_statistics.csv"
		file_cls = dir_base + "{0}/Cluster_statistics.csv"
		#---------------------------------------------------

		for count in clean_args["counts"]:
			#=============== Prepare files & directories ============
			print("Cleaning count: {0}".format(count))
			current = "FGMM_{0}".format(count)
			dir_crr = dir_base + current  + "/"
			os.makedirs(dir_crr,exist_ok=True)

			if count == 0:
				# For the first iteration we copy the filtered members input
				previous = clean_args["input"]

				cmd_cp   = "cp {0} {1}".format(file_input,
							file_mem.format(current))
				os.system(cmd_cp)

			else:
				# For subsequent iterations merge previous source labels with new member data.
				previous = "FGMM_{0}".format(count-1)

				#---------- Read members & classification ----------
				df_src = pd.read_csv(file_src.format(previous),
								usecols=["source_id","label"])
				df_src.set_index(["source_id","label"],inplace=True)
				df_mem = pd.read_csv(file_input)
				df_mem.set_index("source_id",inplace=True)
				#---------------------------------------------------

				#----------- Drop field and label from previous results (we only want cluster labels to remain)
				df_src.drop(index="Field",level="label",
							inplace=True,errors="ignore")
				#----------------------------------------

				#----------------- Merge ----------------------
				df = pd.merge(left=df_src,right=df_mem,
							left_index=True,right_index=True,
							how="left")
				#----------------------------------------------

				#----------- Save merged file ------------
				df.to_csv(file_mem.format(current))
				#----------------------------------
			#=============================================================

			#======================== Infer model ======================================
			if not os.path.exists(file_src.format(current)):
				#--------- Initialize the inference module (Kalkayotl wrapper) --------------------------
				kal = self.Inference(
								dimension=6,
								dir_out=dir_crr,
								zero_points=self.zero_points.copy(),
								indep_measures=False,
								reference_system=self.reference_system,
								sampling_space=self.kalkayotl_args["sampling_space"],
								velocity_model=self.kalkayotl_args["velocity_model"])
				#--------------------------------------------------------------------

				#-------- Load the data set ------------------------------------------
				# It will use the Gaia column names by default.
				kal.load_data(file_mem.format(current),
							sky_error_factor=self.kalkayotl_args["sky_error_factor"])
				#---------------------------------------------------------------------

				# Prepare prior and parameters for the run
				kal.setup(prior="FGMM",
						  parameters=parameters,
						  hyper_parameters=hyper_parameters,
						  parameterization=parameterization)
				# Execute sampling/inference with the configured options
				kal.run(
						tuning_iters=clean_args["tuning_iters"],
						sample_iters=clean_args["sample_iters"],
						target_accept=clean_args["target_accept"],
						chains=clean_args["chains"],
						cores=clean_args["cores"],
						init_iters=clean_args["init_iters"],
						init_refine=clean_args["init_refine"],
						prior_predictive=clean_args["prior_predictive"],
						prior_iters=clean_args["prior_iters"],
						progressbar=clean_args["progressbar"],
						nuts_sampler=clean_args["nuts_sampler"],
						random_seed=clean_args["random_seed"])
				kal.load_trace()
				kal.convergence()
				kal.plot_chains()
				kal.plot_prior_check()
				kal.plot_model(chain=1)
				kal.save_statistics(hdi_prob=self.kalkayotl_args["hdi_prob"])
			#=========================================================================
			# Note: subsequent code assumes the existence of CSVs created by Kalkayotl
			# (Cluster_statistics.csv and Sources_statistics.csv). Careful with file paths.

			#================ Verify convergence and minimum weight ===================
			#-------- Read statistics ---------------------
			df_cls = pd.read_csv(file_cls.format(current))
			df_cls.set_index("Parameter",inplace=True)
			df_src = pd.read_csv(file_src.format(current))
			df_src.set_index("source_id",inplace=True)
			#---------------------------------------------

			#------------- Assess convergence ---------------------------------
			mask = np.isfinite(df_cls["r_hat"])
			condition = df_cls.loc[mask,"r_hat"] > clean_args["conv_r_hat"]
			not_converged = np.any(condition)
			if not_converged:
				print(df_cls.loc[condition,"r_hat"])
				sys.exit("The {0} iteration did not converged!".format(current))
			else:
				print("The {0} iteration converged!".format(current))
			#-----------------------------------------------------------------

			#---------- Stop if the field is negligible ---------------
			field_fraction = df_cls.loc["6D::weights[Field]","mean"]
			negl_field = field_fraction < clean_args["min_field_frac"]
			ids_field = df_src.loc[df_src["label"] == "Field"].index.values
			if negl_field or len(ids_field) == 0:
				print("{0} has negligible field weight".format(current))

				#----------- Drop field and label -------
				df_src = df_src.loc[:,["source_id","label"]]
				df_src.set_index(["source_id","label"],inplace=True)
				df_src.drop(index="Field",level="label",
							inplace=True,errors="ignore")
				#----------------------------------------

				#----------------- Merge ----------------------
				df = pd.merge(left=df_src,right=df_mem,
							left_index=True,right_index=True,
							how="left")
				#----------------------------------------------

				#----------- Save file ------------
				df.to_csv(self.file_cln_mem)
				#----------------------------------
				break
			#------------------------------------------------------------
			#==============================================================================

	def run_kalkayotl(self,
		distributions = ["Gaussian","StudentT","CGMM"],
		args={},
		):
		"""
		Wrapper that runs Kalkayotl inference for a suite of distribution types.
		For CGMM distributions, it creates variants with different numbers of
		mixture components according to args['min_gmm_components']..['max_gmm_components'].
		"""
		#============== Models list construction =========================================
		list_of_distributions = [
			{"name":"Gaussian",
			"parameters":{"location":None,"scale":None},
			"hyper_parameters":{
								"location":None,
								"scale":None,
								"eta":None,
								}
			},
			{"name":"StudentT",
			"parameters":{"location":None,"scale":None,"nu":None},
			"hyper_parameters":{
								"location":None,
								"scale":None,
								"eta":None,
								"nu":None,
								}
			}
			]
		for n_components in range(args["min_gmm_components"],args["max_gmm_components"]+1):
			list_of_distributions.append(
				{"name":"CGMM",
				"parameters":{"location":None,"scale":None,"weights":None},
				"hyper_parameters":{
									"location":None,
									"scale":None,
									"gamma":None,
									"weights":{"n_components":n_components},
									"eta":None,
									}
			})
		#====================================================================

		# Copy and adjust zero_points dictionary (keeps same key)
		zero_points = self.zero_points.copy()
		zero_points["radial_velocity"] = zero_points.pop("radial_velocity")

		#--------------------- Loop over distributions types ------------------------------------
		for distribution in list_of_distributions:

			if distribution["name"] not in distributions:
				continue

			#------ Output directories for each prior -----------------------------------
			dir_out = self.dir_main +"/Kalkayotl/"+ distribution["name"]
			#-------------------------------------------------------------------------

			#---------- Continue if file already present ----------
			if os.path.exists(dir_out+"/Cluster_statistics.csv"):
				continue
			#------------------------------------------------------

			#----- Create model directory -------
			os.makedirs(dir_out,exist_ok=True)
			#------------------------------------

			#--------- Initialize the inference module (Kalkayotl) ------------------------
			kal = self.Inference(dimension=6,
							dir_out=dir_out,
							zero_points=zero_points,
							indep_measures=False,
							reference_system=self.reference_system,
							sampling_space=args["sampling_space"],
							velocity_model=args["velocity_model"])

			#-------- Load the cleaned members data set ------------------------------------
			kal.load_data(self.file_cln_mem,
						sky_error_factor=args["sky_error_factor"])
			#------ Prepares the model -------------------
			kal.setup(prior=distribution["name"],
					  parameters=distribution["parameters"],
					  hyper_parameters=distribution["hyper_parameters"],
					  parameterization=args["parameterization"])

			# Run the full inference with provided options and save statistics/plots.
			kal.run(
					tuning_iters=args["tuning_iters"],
					sample_iters=args["sample_iters"],
					target_accept=args["target_accept"],
					chains=args["chains"],
					cores=args["cores"],
					step=args["step"],
					step_size=args["step_size"],
					init_method=args["init_method"],
					init_iters=args["init_iters"],
					init_absolute_tol=args["init_absolute_tol"],
					init_relative_tol=args["init_relative_tol"],
					init_plot_iters=args["init_plot_iters"],
					init_refine=args["init_refine"],
					prior_predictive=args["prior_predictive"],
					nuts_sampler=args["nuts_sampler"],
					random_seed=args["random_seed"],
					)

			kal.load_trace()
			kal.convergence()
			kal.plot_chains()
			kal.plot_prior_check()
			kal.plot_model(chain=1)
			kal.save_statistics(hdi_prob=args["hdi_prob"])

	def run(self,iterations,
		synthetic_seeds=[0,1,2,3,4,5,6,7,8,9],
		bins = [4.0,6.0,8.0,10.0,12.0,14.0,16.0,18.0,20.0],
		covariate_limits = [4.0,22.0],
		n_cluster_real=int(1e3),
		n_field_real=int(1e3),
		n_samples_syn=int(1e3),
		minimum_nmin=100,
		init_min_det=1e-3,
		best_model_criterion="AIC",
		replace_probabilities=False,
		use_prior_probabilities=False,
		chunks=10
		):
		"""
		High level driver to run several iterations (a loop) that:
		- sets up iter_* directories
		- runs Kalkayotl cleaning/inference
		- runs real data modeling and synthetic experiments
		- computes thresholds and selects new members iteratively
		"""
		base = self.dir_base + "/iter_{0}"
		base_members = base + "/Classification/members_mecayotl.csv"

		print(30*"="+" START "+30*"=")
		for iteration in range(0,iterations):
			print(30*"-"+" Iteration {0} ".format(iteration)+30*"-")

			# If this iteration already produced members, skip it.
			if os.path.exists(base_members.format(iteration)):
				continue

			#--------- Initialization -----------------------------------
			self.initialize_directories(dir_main=base.format(iteration))
			self.best_gmm  = {}
			#-----------------------------------------------------------

			#-------------- First iteration -------------------
			if iteration == 0:
				file_members = self.file_members
			else:
				# Use previous iteration's selected members as input for this iteration.
				file_members = base_members.format(iteration-1)

				#--------- Copy field models from previous iteration (shell cp) -------------------
				cmd = 'cp {0}/Real/Models/Field* {1}/Real/Models/'.format(
					base.format(iteration-1),base.format(iteration))
				os.system(cmd)
				#--------------------------------------------------------------
			#----------------------------------------------------

			#----------- Kalkayotl --------------------------------
			# If clean_members outcome missing, run filter_members and clean_members
			if not os.path.exists(self.file_cln_mem):
				self.filter_members(
							file_input=file_members,
							file_output=self.file_filtered_members,
							args=self.members_args)
				self.clean_members(
							file_input=self.file_flt_mem,
							file_output=self.file_cln_mem,
							args=self.clean_args)

			self.run_kalkayotl(
				distributions=self.kalkayotl_args["distribution"],
				args=self.kalkayotl_args)
			#-------------------------------------------------------

			#--------------- Real -------------------------------
			self.run_real(file_catalogue=self.file_gaia,
				file_members=file_members,
				n_cluster=n_cluster_real,
				n_field=n_field_real,
				best_model_criterion=best_model_criterion,
				replace_probabilities=replace_probabilities,
				use_prior_probabilities=use_prior_probabilities,
				minimum_nmin=minimum_nmin,
				init_min_det=init_min_det,
				chunks=chunks,
				)
			#----------------------------------------------------

			#---------- Synthetic -------------------------------
			self.run_synthetic(
				seeds=synthetic_seeds,
				n_cluster=n_samples_syn,
				chunks=chunks,
				replace_probabilities=replace_probabilities,
				use_prior_probabilities=use_prior_probabilities
				)

			self.find_probability_threshold(
				seeds=synthetic_seeds,
				bins=bins,
				covariate_limits=covariate_limits,
				plot_log_scale=True
				)
			#------------------------------------------------------

			#------------- New members -----------------------------
			self.select_members(instance="Real")
			#-------------------------------------------------------
		print(20*"="+" END "+20*"=")



if __name__ == "__main__":
	#----------------- Example usage block ------------------------
	# The main guard shows an example configuration to run Mecayotl for a single
	# cluster (Rup147 in this example). For production use, override file paths
	# and parameters as appropriate.
	dir_repos = "/home/jolivares/Repos/"
	dir_cats  = "/home/jolivares/OCs/Rup147/Mecayotl/catalogues/"
	dir_base  = "/home/jolivares/OCs/Rup147/Mecayotl/"
	#-------------------------------------------------------

	#----------- Files --------------------------------------------
	file_gaia     = dir_cats + "Rup147_SNR3.fits"
	file_members  = dir_cats + "Rup147_members.csv"
	#--------------------------------------------------------------

	members_args = {
		"g_mag_limit":22.0,
		"rv_error_limits":[0.1,2.0],
		"ruwe_threshold":1.4,
		"prob_threshold":0.999936,
		"rv_sd_clipping":3.0,
		"allow_rv_missing":False,
		}

	kalkayotl_args = {
	"distribution":"CGMM"
	}

	isochrones_args = {
		"log_age": 9.3,
		"metallicity":0.012,
		"Av": 0.0,
		"mass_limits":[0.11,1.7],
		"bands":["G","BP","RP"],
		"mass_prior":"Uniform"
		}


	mcy = Mecayotl(
			dir_base=dir_base,
			file_gaia=file_gaia,
			file_members=file_members,
			members_args=members_args,
			kalkayotl_args=kalkayotl_args,
			isochrones_args=isochrones_args,
			nc_cluster=[1],
			nc_field=[1],
			path_ayome=dir_repos+"Ayome/",
			path_kalkayotl=dir_repos+"Kalkayotl/",
			reference_system="Galactic",
			seed=12345)

	mcy.run(
		iterations=1,
		synthetic_seeds=[0],
		n_cluster_real=int(1e3),
		n_field_real=int(1e3),
		n_samples_syn=int(1e3)
		)
