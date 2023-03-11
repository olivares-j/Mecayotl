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

from pygaia.astrometry.vectorastrometry import phase_space_to_astrometry
from pygaia.astrometry.constants import au_km_year_per_sec,au_mas_parsec

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm
from matplotlib import lines as mlines
from matplotlib.patches import Ellipse
from matplotlib.colors import Normalize,TwoSlopeNorm
from tqdm import tqdm

#---------- Configure simbad --------------
Simbad.remove_votable_fields('coordinates')
Simbad.add_votable_fields('velocity')
#------------------------------------------

def get_principal(sigma,level=2.0):
    sd_x   = np.sqrt(sigma[0,0])
    sd_y   = np.sqrt(sigma[1,1])
    rho_xy = sigma[0,1]/(sd_x*sd_y)


    # Author: Jake VanderPlas
    # License: BSD
    #----------------------------------------
    sigma_xy2 = rho_xy * sd_x * sd_y

    alpha = 0.5 * np.arctan2(2 * sigma_xy2,(sd_x ** 2 - sd_y ** 2))
    tmp1  = 0.5 * (sd_x ** 2 + sd_y ** 2)
    tmp2  = np.sqrt(0.25 * (sd_x ** 2 - sd_y ** 2) ** 2 + sigma_xy2 ** 2)

    return level*np.sqrt(tmp1 + tmp2), level*np.sqrt(np.abs(tmp1 - tmp2)), alpha* 180. / np.pi

class Mecayotl(object):
	"""
	Mecayotl is an algorithm desgined to identify members of open clusters.
	Mecayotl means: genealogia o parentesco.
	https://gdn.iib.unam.mx/termino/search?queryCreiterio=mecayotl&queryPartePalabra=cualquiera&queryBuscarEn=nahuatl&queryLimiteRegistros=50 
	"""

	def __init__(self,dir_main,photometric_args,
					nc_cluster=range(2,21),
					nc_field=range(2,21),
					path_mcmichael = "/home/jolivares/Repos/McMichael/",
					path_amasijo   = "/home/jolivares/Repos/Amasijo/",
					path_kalkayotl = "/home/jolivares/Repos/Kalkayotl/",
					cmap_probability="viridis_r",
					cmap_features="viridis_r",
					zero_point=[0.,0.,-0.017,0.,0.,0.],
					use_GPU=False,
					rv_names={"rv":"dr3_radial_velocity",
							  "rv_error":"dr3_radial_velocity_error"},
					seed=1234):

		gaia_observables = ["source_id",
		"ra","dec","parallax","pmra","pmdec",rv_names["rv"],
		"ra_error","dec_error","parallax_error","pmra_error","pmdec_error",rv_names["rv_error"],
		"ra_dec_corr","ra_parallax_corr","ra_pmra_corr","ra_pmdec_corr",
		"dec_parallax_corr","dec_pmra_corr","dec_pmdec_corr",
		"parallax_pmra_corr","parallax_pmdec_corr",
		"pmra_pmdec_corr",
		"g","rp"]

		#------ Set Seed -----------------
		np.random.seed(seed=seed)
		self.random_state = seed#np.random.RandomState(seed=seed)
		self.seed = seed
		#----------------------------------------------------

		#------------- Repo paths -----------------
		self.path_mcmi       = path_mcmichael
		self.path_amasijo    = path_amasijo
		self.path_kalkayotl  = path_kalkayotl
		#-------------------------------------------------

		#--------- Directories ---------------------------
		self.dir_main  = dir_main
		self.dir_kal   = dir_main + "Kalkayotl"
		#------------------------------------------

		#---------------- Files --------------------------------------------------
		self.file_mem_kal    = self.dir_kal + "/members+rvs.csv"
		self.file_mod_kal    = self.dir_kal + "/{0}_central/Cluster_statistics.csv"
		self.file_smp_base   = dir_main + "/{0}/Data/members_synthetic.csv"
		self.file_data_base  = dir_main + "/{0}/Data/data.h5"
		self.file_model_base = dir_main + "/{0}/Models/{1}_GMM_{2}.h5"
		self.file_comparison = dir_main + "/{0}/Models/{1}_comparison.png"
		self.file_qlt_base   = dir_main + "/Classification/quality_{0}_{1}.{2}"
		self.file_mem_data   = dir_main + "/Classification/members_mecayotl.csv"
		self.file_mem_plot   = dir_main + "/Classification/members_mecayotl.pdf"
		#-------------------------------------------------------------------------

		#-------------- Parameters -----------------------------------------------------
		self.zero_point= np.array(zero_point)
		self.cmap_prob = plt.get_cmap(cmap_probability)
		self.cmap_feat = plt.get_cmap(cmap_features)
		self.idxs      = [[0,1],[2,1],[0,2],[3,4],[5,4],[3,5]]
		self.plots     = [
						  ["ra","dec"],["pmra","pmdec"],
						  ["parallax","pmdec"],["g_rp","g"],["g_rp","G"]
						 ]
		self.IDS       = gaia_observables[0]
		self.OBS       = gaia_observables[1:7]
		self.UNC       = gaia_observables[7:13]
		self.RHO       = gaia_observables[13:23]
		self.EXT       = gaia_observables[23:]
		self.PRO       = "prob_cls"
		self.nc_case   = {"Field":nc_field,"Cluster":nc_cluster}
		self.best_gmm  = {}
		self.best_kal  = None
		self.photometric_args = photometric_args
		self.use_GPU   = use_GPU
		self.observables = gaia_observables
		self.rv_names  = rv_names
		#----------------------------------------------------------------------------------

		#----- Creal real data direcotries -----
		os.makedirs(dir_main + "/Real",exist_ok=True)
		os.makedirs(dir_main + "/Real/Data",exist_ok=True)
		os.makedirs(dir_main + "/Real/Models",exist_ok=True)
		os.makedirs(self.dir_kal,exist_ok=True)
		#-------------------------------------------

		#----- Initialize Amasijo -------
		sys.path.append(self.path_amasijo)
		#--------------------------------

	def _initialize_mcmichael(self):
		if self.use_GPU:
			#-------------- Commands to replace dimension -----------------------------
			cmd = 'sed -e "s|DIMENSION|{0}|g"'.format(6)
			cmd += ' {0}GPU/Functions_base.py > {0}GPU/Functions.py'.format(self.path_mcmi)
			os.system(cmd)
			#--------------------------------------------------------------------------
		sys.path.append(self.path_mcmi)
		
	def generate_true_cluster(self,file_kalkayotl,n_cluster=int(1e5),instance="Real"):
		"""
		Generate synthetic data based on Kalkayotl input parameters
		"""
		#-------- Libraries --------
		from Amasijo import Amasijo
		#---------------------------

		file_smp  = self.file_smp_base.format(instance)

		#----------- Generate true astrometry ---------------
		ama = Amasijo(kalkayotl_file=file_kalkayotl,
					  photometric_args=self.photometric_args,
					  seed=self.seed)

		X = ama._generate_phase_space(n_stars=n_cluster)

		df_as,_ = ama._generate_true_astrometry(X)
		#----------------------------------------------------

		#----- Rename columns ---------------------------
		df_as.rename(columns={
			ama.labels_true_as[0]:self.observables[1],#"ra",
			ama.labels_true_as[1]:self.observables[2],#"dec",
			ama.labels_true_as[2]:self.observables[3],#"parallax",
			ama.labels_true_as[3]:self.observables[4],#"pmra",
			ama.labels_true_as[4]:self.observables[5],#"pmdec",
			ama.labels_true_as[5]:self.observables[6],#"dr3_radial_velocity"
			},inplace=True)
		#------------------------------------------------

		df_as.to_csv(file_smp,index_label="source_id")

	def assemble_data(self,file_catalogue,file_members,
					n_field=int(1e5),
					instance="Real"):
		#------------ Files ------------------------------
		file_data = self.file_data_base.format(instance)
		file_smp  = self.file_smp_base.format(instance)
		#-------------------------------------------------

		#--------------- Catalogue ---------------------
		print("Reading catalogue ...")
		cat = Table.read(file_catalogue, format='fits')
		df_cat = cat.to_pandas()
		del cat
		n_sources = df_cat.shape[0]
		df_cat = df_cat[self.observables]
		#-----------------------------------------------

		#--------- Members ------------------------------------------
		print("Reading members ...")
		if '.csv' in file_members:
			df_mem = pd.read_csv(file_members)
		elif ".fits" in file_members:
			dat = Table.read(file_members, format='fits')
			df_mem  = dat.to_pandas()
			del dat
		else:
			sys.exit("Format file not recognized. Only CSV of FITS")
		df_mem = df_mem[self.observables]
		#-------------------------------------------------------------

		#----------- Synthetic --------------------------------------
		print("Reading synthetic ...")
		mu_syn    = pd.read_csv(file_smp,usecols=self.OBS).to_numpy()
		sg_syn    = np.zeros((len(mu_syn),6,6))
		#-------------------------------------------------------------

		print("Assembling data ...")

		#------ Extract ---------------------------
		id_data = df_cat.loc[:,self.IDS].to_numpy()
		mu_data = df_cat.loc[:,self.OBS].to_numpy()
		sd_data = df_cat.loc[:,self.UNC].to_numpy()
		cr_data = df_cat.loc[:,self.RHO].to_numpy()
		ex_data = df_cat.loc[:,self.EXT].to_numpy()
		#------------------------------------------

		#---- Substract zero point---------
		mu_data = mu_data - self.zero_point
		#----------------------------------

		#----- Select members and field ----------
		ids_all  = df_cat["source_id"].to_numpy()
		ids_mem  = df_mem["source_id"].to_numpy()
		mask_mem = np.isin(ids_all,ids_mem)
		idx_cls  = np.where(mask_mem)[0]
		idx_fld  = np.where(~mask_mem)[0]
		#-----------------------------------------

		#-------------- Members -----------------------------
		assert len(idx_cls) > 1, "Error: Empty members file!"
		#----------------------------------------------------

		#---------- Random sample of field sources ------------------
		idx_rnd  = np.random.choice(idx_fld,size=n_field,replace=False)
		mu_field = mu_data[idx_rnd]
		#------------------------------------------------------------

		#------------- Write -----------------------------------------
		print("Saving data ...")
		with h5py.File(file_data, 'w') as hf:
			hf.create_dataset('ids',        data=id_data)
			hf.create_dataset('mu',         data=mu_data)
			hf.create_dataset('sd',         data=sd_data)
			hf.create_dataset('cr',         data=cr_data)
			hf.create_dataset('ex',         data=ex_data)
			hf.create_dataset('mu_Cluster', data=mu_syn)
			hf.create_dataset('sg_Cluster', data=sg_syn)
			hf.create_dataset('mu_Field',   data=mu_field)
			hf.create_dataset('idx_Field',  data=idx_fld)
			hf.create_dataset('idx_Cluster',data=idx_cls)
		#-------------------------------------------------------------

		#-------- Clear memory -------------------------
		del df_cat,mu_data,mu_field,mu_syn,sg_syn
		#-----------------------------------------------

		#============= Covariance matrices =====================
		print("Filling covariance matrices ...")
		sg_data = np.zeros((n_sources,6,6))

		#----- There is no correlation with r_vel ---
		idx_tru = np.triu_indices(6,k=1)
		idi     = np.where(idx_tru[1] != 5)[0]
		idx_tru = (idx_tru[0][idi],idx_tru[1][idi])
		#--------------------------------------------

		#-------- sd to diag ------------------
		stds = np.zeros((n_sources,6,6))
		diag = np.einsum('...jj->...j',stds)
		diag[:] = sd_data
		#--------------------------------------

		one = np.eye(6)
		pbar = tqdm(total=n_sources,miniters=10000)
		for i,(sd,corr) in enumerate(zip(stds,cr_data)):
			#------- Correlation ---------
			rho  = np.zeros((6,6))
			rho[idx_tru] = corr
			rho  = rho + rho.T + one
			#-----------------------------

			#---------- Covariance ----------------------
			sg_data[i] = sd.dot(rho.dot(sd))
			#--------------------------------------------

			pbar.update()
		pbar.close()
		#========================================================

		#----- Field sources ------
		sg_field = sg_data[idx_rnd]
		#--------------------------

		#---------------- Write ---------------------------------------
		print("Saving covariance matrices ...")
		with h5py.File(file_data, 'a') as hf:
			hf.create_dataset('sg',      data=sg_data)
			hf.create_dataset('sg_Field',data=sg_field)
		#--------------------------------------------------------------
		del sg_data,sg_field
		print("Data correctly assembled")

	def infer_models(self,case="Field",instance="Real",
					tolerance=1e-5,init_min_det=1e-2):

		#---------- Libraries ------------------
		self._initialize_mcmichael()
		if self.use_GPU:
			from GPU.gmm import GaussianMixture
		else:
			from CPU.gmm import GaussianMixture
		#---------------------------------------

		file_data = self.file_data_base.format(instance)

		#------------ Read data------------------
		print("Reading data ...")
		with h5py.File(file_data, 'r') as hf:
			X = np.array(hf.get("mu_" + case))
			U = np.array(hf.get("sg_" + case))
		#----------------------------------------

		#-- Dimensions --
		N,_ = X.shape
		#----------------

		#-------------------- Loop over models ----------------------------------
		for n_components in self.nc_case[case]:
			file_model = self.file_model_base.format(instance,case,n_components)

			if os.path.isfile(file_model):
				continue

			#------------ Inference ---------------------------------------------
			print("Inferring model with {0} components.".format(n_components))
			gmm = GaussianMixture(dimension=6,n_components=n_components)
			gmm.setup(X,uncertainty=U)
			gmm.fit(tol=tolerance,init_min_det=init_min_det,
				random_state=self.random_state)
			#--------------------------------------------------------------------

			#------- Write --------------------------------
			with h5py.File(file_model,'w') as hf:
				hf.create_dataset('G',    data=n_components)
				hf.create_dataset('pros', data=gmm.weights_)
				hf.create_dataset('means',data=gmm.means_)
				hf.create_dataset('covs', data=gmm.covariances_)
				hf.create_dataset('aic',  data=gmm.aic)
				hf.create_dataset('bic',  data=gmm.bic)
				hf.create_dataset('nmn',  data=N*np.min(gmm.weights_))
			#------------------------------------------------
			print("------------------------------------------")

		del X,U


	def select_best_model(self,case="Field",instance="Real",
							minimum_nmin=100,criterion="AIC"):

		file_comparison = self.file_comparison.format(instance,case)

		aics = []
		bics = []
		nmin = []

		#--------------- Read models ---------------------------
		for n_components in self.nc_case[case]:
			file_model = self.file_model_base.format(instance,case,n_components)
			#--------- Read -------------------------------------------
			with h5py.File(file_model, 'r') as hf:
				aics.append(np.array(hf.get('aic')))
				bics.append(np.array(hf.get('bic')))
				nmin.append(np.array(hf.get('nmn')))
			#-----------------------------------------------------------

		#---- Arrays --------
		aics = np.array(aics)
		bics = np.array(bics)
		nmin = np.array(nmin)
		#---------------------

		#------ Find best ---------------------------------
		idx_valid = np.where(nmin > minimum_nmin)[0]
		if criterion == "BIC":
			idx_min   = np.argmin(bics[idx_valid])
		elif criterion == "AIC":
			idx_min   = np.argmin(aics[idx_valid])
		else:
			sys.exit("Criterion {0} not valid".format(criterion))
		idx_best  = idx_valid[idx_min]
		#--------------------------------------------------

		#------------ Set best model ------------------------------------------
		if instance in self.best_gmm.keys():
			self.best_gmm[instance].update({case:self.nc_case[case][idx_best]})
		else:
			self.best_gmm.update({instance:{case:self.nc_case[case][idx_best]}})
		#-----------------------------------------------------------------------
		
		#-----------Plot BIC,AIC,NMIN ------------------------
		plt.figure(figsize=(8,6))
		axl = plt.gca()
		axl.plot(self.nc_case[case],bics,label="BIC")
		axl.plot(self.nc_case[case],aics,label="AIC")
		axl.set_xlabel('Components')
		axl.set_ylabel("Criterion")
		axl.set_xticks(self.nc_case[case])
		axl.legend(loc="upper left")

		axr = axl.twinx()
		axr.plot(self.nc_case[case],nmin,
				ls="--",color="black",label="$N_{min}$")
		axr.set_yscale("log")
		axr.set_ylabel("N_stars in lightest components")
		axr.legend(loc="upper right")

		axr.axvline(x=self.nc_case[case][idx_best],
						linewidth=1, color='gray',ls=":")

		plt.savefig(file_comparison,bbox_inches='tight')
		plt.close()
		#-----------------------------------------------------


	def plot_model(self,n_components=None,case="Field",instance="Real"):
		#--------------- n_components ----------------------------------------
		if n_components is None:
			n_components = self.best_gmm[instance][case]
		assert isinstance(n_components,int), "n_components must be an integer!"
		#----------------------------------------------------------------------

		file_data  = self.file_data_base.format(instance)
		file_model = self.file_model_base.format(instance,case,n_components)

		#---------- Return if file already present ----------
		if os.path.isfile(file_model.replace(".h5",".pdf")):
			return
		#----------------------------------------------------

		#--------- Read model -----------------------------
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

		#------------ Colorbar ----------------------------
		norm_feat = Normalize(vmin=min(weights),vmax=max(weights)) 

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

			#--------- Sources --------------------------
			ax.scatter(x=X[:,idx[0]],y=X[:,idx[1]],
						marker=".",s=1,
						c="gray",zorder=0,
						rasterized=True,
						label="Data")
			#-------------------------------------------

			#---------------- Inferred model ------------------------------
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

		#------ Velocities -----------------------------------------
		fig, axs = plt.subplots(nrows=2,ncols=2,figsize=None)

		for ax,idx in zip([axs[0,0],axs[0,1],axs[1,0]],self.idxs[3:]):
			mxgr = np.ix_(range(n_components),idx,idx)

			#--------- Sources --------------------------
			ax.scatter(x=X[:,idx[0]],y=X[:,idx[1]],
						marker=".",s=1,
						c="gray",zorder=0,
						rasterized=True,
						label="Data")
			#-------------------------------------------

			#---------------- Inferred model -----------------------------------
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

		#---------- Libraries ------------------
		self._initialize_mcmichael()
		if self.use_GPU:
			from GPU.gmm import GaussianMixture
		else:
			from CPU.gmm import GaussianMixture
		#---------------------------------------

		file_data  = self.file_data_base.format(instance)

		#------- Read data----------------------------------
		print("Reading data ...")
		with h5py.File(file_data, 'r') as hf:
			if self.PRO in hf.keys() and not replace:
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

		#-----------------------------------------------------------
		if use_prior:
			ln_prior_ratio = np.log(np.float64(nf)/np.float64(nc))
		else:
			ln_prior_ratio = 0.0
		#-----------------------------------------------------------
	
		#------- Chunks ----------------------------------------
		# Compute partitioning of the input array of size N
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
		#-----------------------------------------------------

		#----- Likelihoods ----------------------------------
		llks = np.zeros((N,2))
		for i,case in enumerate(["Field","Cluster"]):
			n_components = self.best_gmm[instance][case]
			#--------------- File --------------------------
			file_model = self.file_model_base.format(
				instance,case,n_components)
			#------------------------------------------------

			#------------ Read model -------------------
			print("Reading {0} parameters ...".format(case))
			with h5py.File(file_model, 'r') as hf:
				n_components = np.array(hf.get("G"))
				weights      = np.array(hf.get("pros"))
				means        = np.array(hf.get("means"))
				covariances  = np.array(hf.get("covs"))
			#-------------------------------------------

			#------------ Inference ------------------------------------
			print("Computing likelihoods ...")
			gmm = GaussianMixture(dimension=6,n_components=n_components)

			for idx in chunk_idx:
				gmm.setup(mu[idx],uncertainty=sg[idx])
				llks[idx,i] = logsumexp(gmm.log_likelihoods(
										weights,means,covariances),
										axis=1,keepdims=False)
			del gmm
			#-----------------------------------------------------------

		assert np.all(np.isfinite(llks)), "Likelihoods are not finite!"
		del mu,sg
		#--------------------------------------------------------------

		#------- Probability ------------------------------------------------
		print("Computing probabilities ...")
		pc = 1.0/(1.0+np.exp(ln_prior_ratio + llks[:,0] - llks[:,1]))
		del llks
		assert np.all(np.isfinite(pc)), "Probabilities are not finite!"
		assert np.all(pc >= 0.0), "Probabilities are negative!"
		assert np.all(pc <= 1.0), "Probabilities are larger than one!"
		#--------------------------------------------------------------------

		#----------- Cluster probabilities --------------------------------
		print("Cluster probabilities:")
		print("Min:{0:1.4f}, Mean:{1:1.4f}, Median:{2:1.4f}, Max:{3:1.4f}".format(
			np.min(pc[idx_cls]),np.mean(pc[idx_cls]),
			np.median(pc[idx_cls]),np.max(pc[idx_cls])))
		#------------------------------------------------------------------

		#---------- Save probability -------------------------
		print("Saving probabilities ...")
		with h5py.File(file_data, 'a') as hf:
			if replace:
				del hf[self.PRO]
			hf.create_dataset(self.PRO,data=pc)
		#-----------------------------------------------------------

	def generate_synthetic(self,n_cluster=int(1e5),seeds=range(1)):

		#-------- Libraries --------
		from Amasijo import Amasijo
		#---------------------------

		#================= Generate syntehtic data ====================
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
			ama = Amasijo(photometric_args=self.photometric_args,
						  kalkayotl_file=self.file_mod_kal.format(self.best_kal),
						  seed=seed)
			ama.generate_cluster(file_smp,n_stars=n_cluster,
				angular_correlations=None)
			ama.plot_cluster(file_plot=file_smp.replace(".csv",".pdf"))
			del ama
			#----------------------------------------------------------

	def assemble_synthetic(self,seeds=[0]):
		columns    = [ obs for obs in self.observables \
						if obs not in self.RHO]
		local_seeds = seeds.copy()

		#-------------- Check if data exists ----------------------
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

			#-------- Read cluster -----------------------
			df_cls = pd.read_csv(file_smp,usecols=columns)
			n_cls  = len(df_cls)
			#---------------------------------------------

			#----- Select members and field -----------------
			idx_cls  = np.arange(n_cls)
			#------------------------------------------------

			#------ Extract --------------------------
			mu_cls = df_cls.loc[:,self.OBS].to_numpy()
			sd_cls = df_cls.loc[:,self.UNC].to_numpy()
			ex_cls = df_cls.loc[:,self.EXT].to_numpy()
			del df_cls
			#-----------------------------------------

			#----------- Covariance matrices -------
			zeros = np.zeros((len(sd_cls),6,6))
			diag = np.einsum('...jj->...j',zeros)
			diag[:] = np.square(sd_cls)
			sg_cls = zeros.copy()
			del diag,zeros,sd_cls
			#----------------------------------------

			#--------------- Write data ------------------------------------
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

		#------ Read probabilities ---------------
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

		#================= Cluster ==================================
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

			#----- Field index -------------------------------------
			idx_fld  = np.arange(len(idx_cls),len(idx_cls)+n_field)
			#------------------------------------------------------

			#------- Concatenate ---------------------------
			mu_data = np.concatenate((mu_cls,mu_fld),axis=0)
			sg_data = np.concatenate((sg_cls,sg_fld),axis=0)
			ex_data = np.concatenate((ex_cls,ex_fld),axis=0)
			#-----------------------------------------------

			#--------------- Write data ------------------------------------
			with h5py.File(file_data, 'a') as hf:
				hf.create_dataset('mu',         data=mu_data)
				hf.create_dataset('sg',         data=sg_data)
				hf.create_dataset('ex',         data=ex_data)
				hf.create_dataset('idx_Field',  data=idx_fld)
			#----------------------------------------------------------------
			del idx_fld,mu_data,sg_data,ex_data

	def compute_probabilities_synthetic(self,seeds,chunks=1,
							replace=False,use_prior=False):

		for i,seed in enumerate(seeds):
			#------------ File and direcotry ----------------------------
			instance  = "Synthetic_{0}".format(seed)
			dir_model = "{0}/{1}/Models/".format(self.dir_main,instance)
			os.makedirs(dir_model,exist_ok=True)
			#-------------------------------------------------------------

			#-------- Same models as real data ---------------
			self.best_gmm[instance] = self.best_gmm["Real"]
			#-------------------------------------------------------

			#---------------- Copy best models ------------------------
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
		covariate="g",metric="MCC",covariate_limits=None,
		plot_log_scale=False,
		prob_steps={
				0.954499736103642:10, # 2sigma
				0.997300203936740:10, # 3sigma
				0.999936657516334:10, # 4sigma
				0.999999426696856:10 # 5sigma
				# 0.999999998026825:10  # 6sigma
				},
		min_prob=0.682689492137086):

		#-------- Libraries -------------------
		from Quality import ClassifierQuality
		#--------------------------------------

		file_plot = self.file_qlt_base.format(covariate,metric,"pdf")
		file_tex  = self.file_qlt_base.format(covariate,metric,"tex")
		file_thr  = self.file_qlt_base.format(covariate,metric,"pkl")

		os.makedirs(self.dir_main+"/Classification/",exist_ok=True)

		dfs = []
		for seed in seeds:
			print("Reading seed {0}".format(seed))
			#------------ File ----------------------------
			instance  = "Synthetic_{0}".format(seed)
			file_data = self.file_data_base.format(instance)

			#------ Read data  and probabilities -----------------------
			with h5py.File(file_data, 'r') as hf:
				idx_cls  = np.array(hf.get("idx_Cluster"))
				idx_fld  = np.array(hf.get("idx_Field"))
				pc = np.array(hf.get(self.PRO))
				ex_data  = np.array(hf.get("ex"))
			#-------------------------------------------------

			#---------- Class ------------------------
			classs = np.full(len(pc),fill_value=False)
			classs[idx_cls] = True
			#-----------------------------------------

			#------ Create dataframe ----------------
			df = pd.DataFrame(data={"Cluster":classs,
									self.PRO:pc})
			#----------------------------------------

			#-------- Insert -------------------------
			for ex,name in zip(ex_data.T,self.EXT):
				df.insert(loc=2,column=name,value=ex)
			#----------------------------------------

			#-- Append ----
			dfs.append(df)
			#--------------

			del df

		print("Analyzing classifier quality ...")
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

		self.file_thresholds = file_thr

		del clq

	def select_members(self,probability_threshold=None,instance="Real"):

		file_data = self.file_data_base.format(instance)

		#------ Read probabilities --------------------------
		with h5py.File(file_data, 'r') as hf:
			idx_cls = np.array(hf.get("idx_Cluster"))
			ids = np.array(hf.get("ids"),dtype=np.uint64)
			mu = np.array(hf.get("mu"))
			sd = np.array(hf.get("sd"))
			cr = np.array(hf.get("cr"))
			ex = np.array(hf.get("ex"))
			pc = np.array(hf.get(self.PRO))
		#----------------------------------------------------

		#-------- Join data --------------------------------------
		names = sum([self.OBS,self.UNC,self.RHO,self.EXT],[])
		dt = np.hstack((mu,sd,cr,ex))
		df_cat = pd.DataFrame(data=dt,columns=names)
		df_cat.insert(loc=0,column=self.PRO,value=pc)
		df_cat.insert(loc=0,column=self.IDS,value=ids)
		#---------------------------------------------------------

		#----- Members ---------------
		df_mem  = df_cat.iloc[idx_cls]
		#-----------------------------

		#----- Candidates --------------------------------
		print("Selecting candidates ...")
		if isinstance(probability_threshold,float):
			idx_cnd = np.where(pc >= probability_threshold)[0]
			df_cnd  = df_cat.iloc[idx_cnd]

		else:
			#------------- File tresholds -----------------------------------
			file_thresholds = self.file_thresholds if probability_threshold \
						is None else probability_threshold
			#----------------------------------------------------------------

			#----- Load edges and probability thresholds ------
			with open(file_thresholds,'rb') as in_stream: 
				quality = dill.load(in_stream)
			#--------------------------------------------------

			#------- Split data frame into bins ---------------------
			bin_mag = np.digitize(df_cat[quality["covariate"]].values,
						bins=quality["edges"])
			#--------------------------------------------------------

			#------ Bin 0 objects to bin 1----------------
			# Few objects are brighter than the brightest edge so 
			# we use the same probability as the first bin
			bin_mag[np.where(bin_mag == 0)[0]] = 1
			#-------------------------------------

			#----------- Loop over bins ----------------------------
			dfs = []
			for i,threshold in enumerate(quality["thresholds"]):
				#--------- Objects in bin -----------------------
				idx = np.where(bin_mag == i)[0]
				strategy = "Bin {0}".format(i)
				# There are no objects in bin zero
				# so we use it for all objects
				if i == 0: 
					assert len(idx) == 0 ,"Bin 0 is not empty!"
					idx = np.arange(len(df_cat))
					strategy = "All"
				#------------------------------------------------

				#------------- Members ---------------------
				idx_mem = np.where(pc[idx] >= threshold)[0]
				tmp_mem = df_cat.iloc[idx[idx_mem]].copy()
				tmp_mem["Strategy"] = strategy
				#-------------------------------------------

				#------------ Append -------------------------------------------
				dfs.append(tmp_mem)
				#---------------------------------------------------------------

			#--------- Concatenate and extract -----
			df = pd.concat(dfs)
			df_cnd = df[df["Strategy"] != "All"].copy()
			del df_cat,dfs,df
			#-------------------------------------------
		#-------------------------------------------------

		#---------- IDs ------------------------
		ids_cnd = df_cnd[self.IDS].to_numpy()
		ids_mem = df_mem[self.IDS].to_numpy()
		#---------------------------------------

		#-------- Summary----------------------------
		ids_common = np.intersect1d(ids_mem,ids_cnd)
		ids_reject = np.setdiff1d(ids_mem,ids_cnd)
		ids_new    = np.setdiff1d(ids_cnd,ids_mem)
		print("Common: {0}".format(len(ids_common)))
		print("Rejected: {0}".format(len(ids_reject)))
		print("New: {0}".format(len(ids_new)))
		#---------------------------------------------

		#--------- Save candidates ------------------
		df_cnd.to_csv(self.file_mem_data,index=False)
		#--------------------------------------------

		#----------- Color ---------------------------
		df_mem["g_rp"] = df_mem["g"] - df_mem["rp"]
		df_cnd["g_rp"] = df_cnd["g"] - df_cnd["rp"]
		#---------------------------------------------

		#----------- Absoulute magnitude ------------------
		df_mem["G"] = df_mem["g"] + 5.*( 1.0 - 
						np.log10(1000./df_mem["parallax"]))
		df_cnd["G"] = df_cnd["g"] + 5.*( 1.0 - 
						np.log10(1000./df_cnd["parallax"]))
		#--------------------------------------------------

		#--------------------------------------------------------
		pdf = PdfPages(filename=self.file_mem_plot)
		for i,plot in enumerate(self.plots):
			fig = plt.figure()
			ax  = plt.gca()
			#--------- Sources --------------------------
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

			#------- Invert ----------
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
				chunks=1,minimum_nmin=100,
				replace_probabilities=False,
				use_prior_probabilities=False,
				best_model_criterion="AIC"):

		assert self.best_kal is not None, "You need to specify the best model from Kalkayotl!"

		#-------------------- Synthetic --------------------------
		if not os.path.isfile(self.file_smp_base.format("Real")):
			self.generate_true_cluster(n_cluster=n_cluster,
					file_kalkayotl=self.file_mod_kal.format(self.best_kal))
		#---------------------------------------------------------

		#------------- Assemble -----------------------------------
		if not os.path.isfile(self.file_data_base.format("Real")):
			self.assemble_data(file_catalogue=file_catalogue,
								file_members=file_members,
								n_field=n_field,
								instance="Real")
		#------------------------------------------------------
		
		#--------------- Infer models ---------------------
		self.infer_models(case="Field",instance="Real")
		self.infer_models(case="Cluster",instance="Real")
		#-------------------------------------------------

		#------------- Select best models --------------------------
		if "Real" not in self.best_gmm:
			self.select_best_model(case="Field",instance="Real",
									minimum_nmin=minimum_nmin,
									criterion=best_model_criterion)
			self.select_best_model(case="Cluster",instance="Real",
									minimum_nmin=minimum_nmin,
									criterion=best_model_criterion)
			print("The best real GMM models are:")
			print(self.best_gmm)
		#-----------------------------------------------------------

		#----------------- Plot best models ---------------
		self.plot_model(case="Field",instance="Real")
		self.plot_model(case="Cluster",instance="Real")
		#--------------------------------------------------

		#-------- Probabilities --------------------------------
		self.compute_probabilities(instance="Real",
							chunks=chunks,
							replace=replace_probabilities,
							use_prior=use_prior_probabilities)
		#--------------------------------------------------------

	def run_synthetic(self,seeds,
					n_cluster=int(1e5),chunks=1,
					replace_probabilities=False,
					use_prior_probabilities=False):

		#----------- Synthetic data --------------------------------
		self.generate_synthetic(n_cluster=n_cluster,
							   seeds=seeds)
		self.assemble_synthetic(seeds=seeds)
		self.compute_probabilities_synthetic(seeds,
						chunks=chunks,
						replace=replace_probabilities,
						use_prior=use_prior_probabilities)
		#----------------------------------------------------------

	def members_to_kalkayotl(self,file_members,file_apogee,
			g_mag_limit=None,
			rv_error_limits=[0.01,50.], # Bounds for rv error
			ruwe_threshold=1.4,         # Remove stars with higher RUWE
			rv_sd_clipping=1.0):        # Remove outliers

		#----------- Miscelaneous -----------------
		apogee_columns = ["RA","DEC","GAIAEDR3_SOURCE_ID","VHELIO_AVG","VSCATTER","VERR"]
		rename_columns = {"VHELIO_AVG":"apogee_rv","GAIAEDR3_SOURCE_ID":"source_id"}
		input_rv_names = {"rv":"dr3_radial_velocity","rv_error":"dr3_radial_velocity_error"}
		#------------------------------------------

		#=============== APOGEE ===============================
		#----- Load APOGEE ----------------------------------
		apogee = Table.read(file_apogee, format='fits',hdu=1)
		#----------------------------------------------------

		#--- Extract desired columns ----------------
		apogee = apogee[apogee_columns]
		#--------------------------------------------

		#- Transform to pandas DF ---
		apogee = apogee.to_pandas()
		#----------------------------

		#------------- RV error ---------------
		apogee["apogee_rv_error"] = np.where(
			apogee["VSCATTER"] == 0.0, 
			apogee["VERR"],
			apogee["VSCATTER"])
		#--------------------------------------

		#------- Rename columns ---------------------------
		apogee.rename(columns=rename_columns,inplace=True)
		#--------------------------------------------------

		#------ Drop missing RA,DEC ----------------------------
		apogee.dropna(subset=["RA","DEC"],inplace=True)
		#-------------------------------------------------------

		#--------- Drop unused columns ----------------------
		apogee = apogee[["source_id","apogee_rv","apogee_rv_error"]]
		#----------------------------------------------------

		#----- Set index -------------------------
		apogee.drop_duplicates(subset="source_id",inplace=True)
		apogee.set_index("source_id",inplace=True)
		#-----------------------------------------
		#=======================================================

		#============= Load members =========================
		#----- Load catalogue ------------------------
		if '.csv' in file_members:
			df = pd.read_csv(file_members)
		elif ".fits" in file_members:
			dat = Table.read(file_members, format='fits')
			df  = dat.to_pandas()
			del dat
		else:
			sys.exit("Format file not recognized. Only CSV of FITS")
		#-------------------------------------------------------------
		#==============================================================

		#=============== Simbad X-Match =================================
		#----------- Query by name -----------------------------------
		df["Name"] = df.apply(lambda x: "Gaia EDR3 {0}".format(
								np.int_(x["source_id"])),axis=1)

		df_simbad = Simbad.query_objects(df["Name"]).to_pandas()
		#-------------------------------------------------------------

		#---------- Drop redshift values ---------------------------------
		df_simbad.drop(index=df_simbad[df_simbad["RVZ_TYPE"] != "v"].index,
						inplace=True)
		#------------------------------------------------------------------

		#---------- Drop rows with no rv uncertainty ----------------
		df_simbad.dropna(how="any",subset=["RVZ_RADVEL","RVZ_ERROR"],
						inplace=True)
		#-------------------------------------------------------------

		#------- Merge by original query number ---------
		df_simbad.set_index("SCRIPT_NUMBER_ID",inplace=True)
		df = df.merge(df_simbad,left_index=True,
						right_index=True,how="left")
		#-----------------------------------------------

		df.rename(columns={
					"RVZ_RADVEL":"simbad_radial_velocity",
					"RVZ_ERROR":"simbad_radial_velocity_error"},
					inplace=True)

		#--- Assert that observed values have uncertainties and viceversa ----
		nan_rvs = np.isnan(df["simbad_radial_velocity"].values)
		nan_unc = np.isnan(df["simbad_radial_velocity_error"].values)
		np.testing.assert_array_equal(nan_rvs,nan_unc,
		err_msg="Simbad: There are discrepant missing uncertainties and values!")
		#---------------------------------------------------------------------
		#================================================================

		#------- Set index --------------------
		df.set_index("source_id",inplace=True)
		assert df.index.is_unique, "Index values are not unique. Remove duplicated sources!"
		#--------------------------------------

		#------- Drop faint members ---------------
		if g_mag_limit is not None:
			df = df.loc[df["g"] < g_mag_limit]
		#----------------------------------------------
		#================================================

		#============= X-Match APOGEE ===================================
		#----------------- Merge ----------------------------------------
		print("Merging with original catalogue ...")
		df = df.merge(apogee,how="left",left_index=True,right_index=True,
					validate="one_to_one",
					suffixes=["_original","_apogee"],sort=False)
		#----------------------------------------------------------------
		#================================================================

		#----------- Use APOGEE or Gaia or Simbad when available -----------------------
		df["radial_velocity"] = df.apply(lambda x: x["apogee_rv"]
								if np.isfinite(x["apogee_rv"]) 
								else x[input_rv_names["rv"]]
								if np.isfinite(x[input_rv_names["rv"]]) 
								else x["simbad_radial_velocity"],
								axis=1)
		df["radial_velocity_error"] = df.apply(lambda x: x["apogee_rv_error"]  
								if np.isfinite(x["apogee_rv"]) 
								else x[input_rv_names["rv_error"]]
								if np.isfinite(x[input_rv_names["rv_error"]]) 
								else x["simbad_radial_velocity_error"],
								axis=1)
		#-------------------------------------------------------------------------------

		#--- Assert that observed values have uncertainties and viceversa ----
		nan_rvs = np.isnan(df["radial_velocity"].values)
		nan_unc = np.isnan(df["radial_velocity_error"].values)
		np.testing.assert_array_equal(nan_rvs,nan_unc,
		err_msg="There are discrepant rvs missing uncertainties and values!")
		#---------------------------------------------------------------------

		print("Replacing minumum and maximum uncertainties ...")
		#----------- Set minimum uncertainty -------------------------------------
		condition = df["radial_velocity_error"] < rv_error_limits[0]
		df.loc[condition,"radial_velocity_error"] = rv_error_limits[0]
		#-------------------------------------------------------------------------

		#----------- Set maximum uncertainty -------------------------------------
		condition = df["radial_velocity_error"] > rv_error_limits[1]
		df.loc[condition,"radial_velocity"] = np.nan
		df.loc[condition,"radial_velocity_error"]  = np.nan
		#-------------------------------------------------------------------------

		#------------- Binaries -------------------------------
		condition = df.loc[:,"ruwe"] > ruwe_threshold
		df.loc[condition,"radial_velocity"] = np.nan
		df.loc[condition,"radial_velocity_error"]  = np.nan
		print("Binaries: {0}".format(sum(condition)))
		#-----------------------------------------------------

		#---------- Outliers --------------------------------------------------------
		mu_rv = np.nanmean(df["radial_velocity"])
		sd_rv = np.nanstd(df["radial_velocity"])
		print("Radial velocity: {0:2.1f} +/- {1:2.1f} km/s".format(mu_rv,sd_rv))
		maha_dst = np.abs(df["radial_velocity"] - mu_rv)/sd_rv
		condition = maha_dst > rv_sd_clipping
		df.loc[condition,"radial_velocity"] = np.nan
		df.loc[condition,"radial_velocity_error"]  = np.nan
		print("Outliers: {0}".format(sum(condition)))
		#----------------------------------------------------------------------------

		print("Saving file ...")
		#------- Save as csv ---------
		df.to_csv(self.file_mem_kal)
		#-----------------------------
		del df
		#==================================================================

	def run_kalkayotl(self,
		gmm_n = 2,
		tuning_iters = 3000,
		sample_iters = 3000,
		target_accept = 0.95,
		optimize = True,
		hdi_prob = 0.95
		):

		#----- Import the module -----------------
		sys.path.append(self.path_kalkayotl)
		from kalkayotl.inference import Inference
		#-----------------------------------------

		#============== Prior ===============================================
		list_of_prior = [
			# {"type":"Gaussian",
			# 	"parameters":{"location":None,"scale":None},
			# 	"hyper_parameters":{
			# 						"alpha":None,
			# 						"beta":50.0,
			# 						"gamma":None,
			# 						"delta":None,
			# 						"eta":None
			# 						},
			# 	"parametrization":"central",
			# },

			{"type":"CGMM",      
				"parameters":{"location":None,"scale":None,"weights":None},
				"hyper_parameters":{
									"alpha":None,
									"beta":50.0, 
									"gamma":None,
									"delta":np.repeat(2,gmm_n),
									"eta":None,
									"n_components":gmm_n
									},
				"field_sd":None,
				"parametrization":"central",
				"velocity_model":"joint",
			},
			]
		#====================================================================

		#--------------------- Loop over prior types ------------------------------------
		for prior in list_of_prior:

			#------ Output directories for each prior --------------------------------
			dir_prior = self.dir_kal +"/"+ prior["type"] + "_" + prior["parametrization"]
			#-------------------------------------------------------------------------

			#---------- Continue if file already present ----------
			if os.path.exists(dir_prior+"/Cluster_statistics.csv"):
				continue
			#------------------------------------------------------

			#----- Create prior directory -------
			os.makedirs(dir_prior,exist_ok=True)
			#------------------------------------

			#--------- Initialize the inference module ----------------------------------------
			kal = Inference(dimension=6,
							dir_out=dir_prior,
							zero_point=self.zero_point,
							indep_measures=False,
							reference_system="Galactic")

			#-------- Load the data set --------------------
			# It will use the Gaia column names by default.
			kal.load_data(self.file_mem_kal)

			#------ Prepares the model -------------------
			kal.setup(prior=prior["type"],
					  parameters=prior["parameters"],
					  hyper_parameters=prior["hyper_parameters"],
					  transformation="pc",
					  parametrization=prior["parametrization"],
					  field_sd=prior["field_sd"],
					  velocity_model=prior["velocity_model"])

			kal.run(sample_iters=sample_iters,
					tuning_iters=tuning_iters,
					target_accept=target_accept,
					optimize=optimize,
					chains=2,
					cores=2)

			kal.load_trace()
			kal.convergence()
			kal.plot_chains()
			kal.plot_model(chain=1)
			kal.save_statistics(hdi_prob=hdi_prob)




if __name__ == "__main__":
	#----------------- Directories ------------------------
	dir_repos = "/home/jolivares/Repos/"
	dir_cats  = "/home/jolivares/OCs/TWH/Mecayotl/catalogues/"
	dir_main  = "/home/jolivares/OCs/TWH/Mecayotl/runs/iter_0/"
	#-------------------------------------------------------

	#----------- Files --------------------------------------------
	file_apogee    = "/home/jolivares/OCs/APOGEE/allStar-dr17-synspec_rev1.fits"
	file_members   = dir_cats + "members_Nuria_GDR3.csv"
	file_catalogue = dir_cats + "TWH_SNR3.fits"
	#--------------------------------------------------------------

	#----- Miscellaneous ------------------------------------------------
	seeds = [0,1,2,3,4,5,6,7,8,9]
	bins  = [2.0,6.0,8.0,10.0,12.0,14.0,16.0,18.0,20.0]
	covariate_limits = [2.0,22.0]
	photometric_args = {
	"log_age": 7.0,    
	"metallicity":0.012,
	"Av": 0.0,         
	"mass_limits":[0.01,2.5], 
	"bands":["V","I","G","BP","RP"],
	"mass_prior":"Uniform"
	}
	#---------------------------------------------------------------------
	

	mcy = Mecayotl(dir_main=dir_main,
			   photometric_args=photometric_args,
			   nc_cluster=range(2,10,1),
			   nc_field=range(2,10,1),
			   use_GPU=False,
			   path_amasijo=dir_repos+"Amasijo/",
			   path_mcmichael=dir_repos+"McMichael/",
			   path_kalkayotl=dir_repos+"Kalkayotl/",
			   seed=12345)

	#----------- Kalkayotl ------------------------------
	# mcy.members_to_kalkayotl(file_members=file_members,
	# 					file_apogee=file_apogee,
	# 					g_mag_limit=22.0,
	# 					rv_error_limits=[0.1,2.],
	# 					ruwe_threshold=1.4,
	# 					rv_sd_clipping=1.0)
	# mcy.run_kalkayotl()
	mcy.best_kal = "Gaussian"
	#-----------------------------------------------------

	#--------------- Real -------------------------------
	mcy.run_real(file_catalogue=file_catalogue,
			file_members=file_members,
			n_cluster=int(1e3),
			n_field=int(1e3),
			chunks=1,
			minimum_nmin=10,
			best_model_criterion="AIC",
			replace_probabilities=False,
			use_prior_probabilities=False)
	mcy.best_gmm = {'Real': {'Field': 4, 'Cluster': 4}}
	#----------------------------------------------------

	#---------- Synthetic -------------------------------
	mcy.run_synthetic(seeds=seeds,
			n_cluster=int(1e3),
			chunks=10,
			replace_probabilities=False,
			use_prior_probabilities=False)

	mcy.find_probability_threshold(seeds=seeds,bins=bins,
					covariate_limits=covariate_limits,
					plot_log_scale=True)
	#------------------------------------------------------

	#------------- New members -----------------------------
	mcy.select_members(instance="Real")
	#-------------------------------------------------------

	


	


