import sys
import os
import numpy as np
import pandas as pd
import h5py
import dill

from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from astropy.table import Table
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

	def __init__(self,file_kalkayotl,photometric_args,path_main,
					nc_cluster=range(2,21),
					nc_field=range(2,21),
					path_mcmichael = "/home/jolivares/Repos/McMichael/",
					path_amasijo   = "/home/jolivares/Repos/Amasijo/",
					cmap_probability="viridis_r",
					cmap_features="viridis_r",
					zero_point=[0.,0.,-0.017,0.,0.,0.],
					seed=1234):

		gaia_observables = [
		"ra","dec","parallax","pmra","pmdec","dr2_radial_velocity",
		"ra_error","dec_error","parallax_error","pmra_error","pmdec_error","dr2_radial_velocity_error",
		"ra_dec_corr","ra_parallax_corr","ra_pmra_corr","ra_pmdec_corr",
		"dec_parallax_corr","dec_pmra_corr","dec_pmdec_corr",
		"parallax_pmra_corr","parallax_pmdec_corr",
		"pmra_pmdec_corr"]

		#------ Set Seed -----------------
		np.random.seed(seed=seed)
		self.random_state = np.random.RandomState(seed=seed)
		self.seed = seed
		#----------------------------------------------------

		#------------- Directories -----------------
		self.dir_mcmi       = path_mcmichael
		self.dir_amasijo    = path_amasijo
		self.path_main      = path_main
		#--------------------------------------

		#---------------- Files --------------------------------------------------
		self.file_kalkayotl  = file_kalkayotl
		self.file_smp_base   = path_main + "/{0}/Data/members_synthetic.csv"
		self.file_hdf_base   = path_main + "/{0}/Data/catalogue.h5"
		self.file_data_base  = path_main + "/{0}/Data/data.h5"
		self.file_model_base = path_main + "/{0}/Models/{1}_GMM_{2}.h5"
		self.file_comparison = path_main + "/{0}/Models/{1}_comparison.png"
		self.file_qlt_base   = path_main + "/Classification/quality_{0}_{1}.{2}"
		self.file_mem_data   = path_main + "/Classification/members_mecayotl.csv"
		self.file_mem_plot   = path_main + "/Classification/members_mecayotl.pdf"
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
		self.OBS       = gaia_observables[0:6]
		self.UNC       = gaia_observables[6:12]
		self.RHO       = gaia_observables[12:]
		self.nc_case   = {"Field":nc_field,"Cluster":nc_cluster}
		self.best_gmm  = {}
		self.photometric_args = photometric_args
		#----------------------------------------------------------------------------------

		#----- Creal real data direcotries -----
		os.makedirs(path_main + "/Real",exist_ok=True)
		os.makedirs(path_main + "/Real/Data",exist_ok=True)
		os.makedirs(path_main + "/Real/Models",exist_ok=True)
		#-------------------------------------------

		#----- Initialize Amasijo -------
		sys.path.append(self.dir_amasijo)
		#--------------------------------

	def _initialize_mcmichael(self):
		#-------------- Commands to replace dimension -----------------------------
		cmd = 'sed -e "s|DIMENSION|{0}|g"'.format(6)
		cmd += ' {0}GPU/Functions_base.py > {0}GPU/Functions.py'.format(self.dir_mcmi)
		os.system(cmd)
		#--------------------------------------------------------------------------
		sys.path.append(self.dir_mcmi)
		
	def generate_true_cluster(self,file_kalkayotl,n_samples=100000,instance="Real"):
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

		X = ama._generate_phase_space(n_stars=n_samples)

		df_as,_ = ama._generate_true_astrometry(X)
		#----------------------------------------------------

		#----- Rename columns ---------------------------
		df_as.rename(columns={
			ama.labels_true_as[0]:"ra",
			ama.labels_true_as[1]:"dec",
			ama.labels_true_as[2]:"parallax",
			ama.labels_true_as[3]:"pmra",
			ama.labels_true_as[4]:"pmdec",
			ama.labels_true_as[5]:"dr2_radial_velocity"
			},inplace=True)
		#------------------------------------------------

		df_as.to_csv(file_smp,index_label="source_id")

	def assemble_data(self,file_catalogue,file_members,
						n_fld=100000,instance="Real"):
		#------------ Files ------------------------------
		file_data = self.file_data_base.format(instance)
		file_hdf  = self.file_hdf_base.format(instance)
		file_smp  = self.file_smp_base.format(instance)
		#-------------------------------------------------

		#--------------- Catalogue ---------------------
		cat = Table.read(file_catalogue, format='fits')
		df_cat = cat.to_pandas()
		del cat
		#-----------------------------------------------

		#--------- Members ------------------------------------------
		if '.csv' in file_members:
			df_mem = pd.read_csv(file_members)
		elif ".fits" in file_members:
			dat = Table.read(file_members, format='fits')
			df_mem  = dat.to_pandas()
			del dat
		else:
			sys.exit("Format file not recognized. Only CSV of FITS")
		#-------------------------------------------------------------

		#----------- Synthetic -----------------------------------
		df_syn = pd.read_csv(file_smp,usecols=self.OBS)
		#---------------------------------------------------

		n_sources = df_cat.shape[0]
		mu_syn    = df_syn.to_numpy()
		sg_syn    = np.zeros((len(mu_syn),6,6))

		#----------- Covariance matrices ------------------
		print("Filling covariance matrices ...")
		sg_data = np.zeros((n_sources,6,6))

		#------ Extract ---------------------------
		mu_data = df_cat.loc[:,self.OBS].to_numpy()
		stds    = df_cat.loc[:,self.UNC].to_numpy()
		cr_data = df_cat.loc[:,self.RHO].to_numpy()
		#------------------------------------------

		#---- Substract zero point---------
		mu_data = mu_data - self.zero_point
		#----------------------------------

		#----- There is no correlation with r_vel ---
		idx_tru = np.triu_indices(6,k=1)
		idi     = np.where(idx_tru[1] != 5)[0]
		idx_tru = (idx_tru[0][idi],idx_tru[1][idi])
		#--------------------------------------------

		#-------- sd to diag ------------------
		sd_data = np.zeros((n_sources,6,6))
		diag = np.einsum('...jj->...j',sd_data)
		diag[:] = stds 
		#--------------------------------------

		one = np.eye(6)
		pbar = tqdm(total=n_sources,miniters=10000)
		for i,(sd,corr) in enumerate(zip(sd_data,cr_data)):
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

		#----- Select members and field ----------
		ids_all  = df_cat["source_id"].to_numpy()
		ids_mem  = df_mem["source_id"].to_numpy()
		mask_mem = np.isin(ids_all,ids_mem)
		idx_cls  = np.where(mask_mem)[0]
		idx_fld  = np.where(~mask_mem)[0]
		#-----------------------------------------

		#-------------- Members -----------------------------
		assert len(idx_cls) > 1, "Error: Empty members file!"
		df_cat["Member"] = False
		df_cat.loc[mask_mem,"Member"] = True
		#----------------------------------------------------
		
		#---------- Random sample of field sources ------------------
		idx_rnd  = np.random.choice(idx_fld,size=n_fld,replace=False)
		mu_field = mu_data[idx_rnd]
		sg_field = sg_data[idx_rnd]
		#------------------------------------------------------------
		
		#------------- Write -----------------------------------------
		df_cat.to_hdf(file_hdf,key="catalogue",mode="w")

		with h5py.File(file_data, 'w') as hf:
			hf.create_dataset('mu',         chunks=True,data=mu_data)
			hf.create_dataset('sg',         chunks=True,data=sg_data)
			hf.create_dataset('mu_Cluster', chunks=True,data=mu_syn)
			hf.create_dataset('mu_Field',   chunks=True,data=mu_field)
			hf.create_dataset('sg_Cluster', chunks=True,data=sg_syn)
			hf.create_dataset('sg_Field',   chunks=True,data=sg_field)
			hf.create_dataset('idx_Field',  chunks=True,data=idx_fld)
			hf.create_dataset('idx_Cluster',chunks=True,data=idx_cls)
		#---------------------------------------------------------
		del df_cat,df_syn,mu_data,sg_data,mu_field,sg_field,mu_syn,sg_syn
		print("Data correctly assembled")

	def infer_models(self,case="Field",instance="Real"):

		#---------- Libraries ------------------
		self._initialize_mcmichael()
		from GPU.gmm import GaussianMixture
		#---------------------------------------

		file_data = self.file_data_base.format(instance)

		#------------ Read data------------------
		print("Reading data ...")
		with h5py.File(file_data, 'r') as hf:
			X = np.array(hf.get("mu_"+case))
			U = np.array(hf.get("sg_"+case))
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
			gmm.fit(random_state=self.random_state)
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
							minimum_nmin=100, criterion="BIC"):

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
			
	def compute_probabilities(self,instance="Real",chunks=1):

		#---------- Libraries ------------------
		self._initialize_mcmichael()
		from GPU.gmm import GaussianMixture
		#---------------------------------------

		file_data  = self.file_data_base.format(instance)
		file_hdf   = self.file_hdf_base.format(instance)

		#------- Read data-------------------
		print("Reading data ...")
		with h5py.File(file_data, 'r') as hf:
			mu = np.array(hf.get("mu"))
			sg = np.array(hf.get("sg"))
		#------------------------------------

		#-- Dimensions --
		N,_ = mu.shape
		#----------------
	

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

		self._initialize_mcmichael()

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

		#------- Probability --------------------------------------
		print("Computing probabilities ...")
		llks = np.exp(llks)
		prob_cls = llks[:,1]/llks.sum(axis=1)
		del llks
		#----------------------------------------------------------

		#---------- Append probability -------------------
		df_cat = pd.read_hdf(file_hdf,key="catalogue",mode="r")
		df_cat["prob_cls"] = prob_cls
		#-----------------------------------------------

		print("Updating catalogue ...")
		df_cat.to_hdf(file_hdf,key="catalogue",format="table",mode="w")
		del df_cat

	def generate_synthetic(self,file_catalogue,
							n_members=1000,
							n_field=100000,
							seeds=range(1)):

		#-------- Libraries --------
		from Amasijo import Amasijo
		#---------------------------

		#================= Generate syntehtic data ====================
		for seed in seeds:
			#--------- Directory ---------------------------------
			name_base = "/Synthetic_{0}/".format(seed)
			dir_sim   = self.path_main + name_base
			file_smp  = self.file_smp_base.format(name_base)
			file_data = self.file_data_base.format(name_base)
			file_hdf  = self.file_hdf_base.format(name_base)
			#----------------------------------------------------

			#--- Create simulation directory -----------
			os.makedirs(dir_sim,exist_ok=True)
			os.makedirs(dir_sim + "Data",exist_ok=True)
			os.makedirs(dir_sim + "Models",exist_ok=True)
			#-------------------------------------------

			# #---------- Generate cluster ---------------------------
			ama = Amasijo(photometric_args=self.photometric_args,
						  kalkayotl_file=self.file_kalkayotl,
						  seed=seed)

			ama.generate_cluster(file_smp,n_stars=n_members)

			ama.plot_cluster(file_plot=file_smp.replace(".csv",".pdf"))
			#----------------------------------------------------------

			#-------- Read cluster and field ---------------
			print("Reading field and cluster data ...")
			df_cls = pd.read_csv(file_smp)
			fld    = Table.read(file_catalogue, format='fits')
			df_fld = fld.to_pandas().sample(n=n_field,
										random_state=seed)
			#-----------------------------------------------

			#------ Rename field radial_velocity -----------------
			df_fld.rename(columns={"radial_velocity":self.OBS[-1],
						   "radial_velocity_error":self.UNC[-1]},
						   inplace=True)
			#-----------------------------------------------------

			#------ Class -----------
			df_cls["Cluster"] = True
			df_fld["Cluster"] = False
			#-------------------------

			#------ Concatenate ------------------------------
			df = pd.concat([df_cls,df_fld],ignore_index=True)
			#------------------------------------------------

			#------ Extract -----------------------
			mu_data = df.loc[:,self.OBS].to_numpy()
			sd_data = df.loc[:,self.UNC].to_numpy()
			#--------------------------------------

			#----- Select members and field -------
			idx_cls  = np.where( df["Cluster"])[0]
			idx_fld  = np.where(~df["Cluster"])[0]
			#--------------------------------------

			#-------- Write & delete ------------------------
			df.to_hdf(file_hdf,key="catalogue",mode="w")
			del df # Release memory
			#----------------------------------------------

			#----------- Covariance matrices ------------------
			print("Filling covariance matrices ...")

			#-------- sd to diag ------------------
			zeros = np.zeros((len(sd_data),6,6))
			diag = np.einsum('...jj->...j',zeros)
			diag[:] = np.square(sd_data)
			sg_data = zeros.copy()
			del diag
			del zeros
			#--------------------------------------

			#--------------- Write data ------------------------------------
			with h5py.File(file_data, 'w') as hf:
				hf.create_dataset('mu',         chunks=True,data=mu_data)
				hf.create_dataset('sg',         chunks=True,data=sg_data)
				hf.create_dataset('mu_Cluster', chunks=True,data=mu_data[idx_cls])
				hf.create_dataset('mu_Field',   chunks=True,data=mu_data[idx_fld])
				hf.create_dataset('sg_Cluster', chunks=True,data=sg_data[idx_cls])
				hf.create_dataset('sg_Field',   chunks=True,data=sg_data[idx_fld])
				hf.create_dataset('idx_Field',  chunks=True,data=idx_fld)
				hf.create_dataset('idx_Cluster',chunks=True,data=idx_cls)
			#----------------------------------------------------------------
			print("Data correctly written")
			del mu_data
			del sg_data
			del sd_data

	def compute_probabilities_synthetic(self,seeds):

		for i,seed in enumerate(seeds):
			#------------ File and direcotry ----------------------------
			instance  = "Synthetic_{0}".format(seed)
			dir_model = "{0}/{1}/Models/".format(self.path_main,instance)
			file_hdf  = self.file_hdf_base.format(instance)
			os.makedirs(dir_model,exist_ok=True)
			#-------------------------------------------------------------
			print("Analyzing {0} data ...".format(instance))

			if i == 0:
				#---------- Infer field model --------------------------
				self.infer_models(case="Field",instance=instance)
				self.select_best_model(case="Field",instance=instance)
				self.plot_model(case="Field",instance=instance)
				#-------------------------------------------------------

				#-------- Best cluster same as Real data ---------------
				self.best_gmm[instance].update({"Cluster":
									self.best_gmm["Real"]["Cluster"]})
				#-------------------------------------------------------

			else:
				#---------------- Copy field models ------------------------
				model_fld = self.file_model_base.format(
					"Synthetic_{0}".format(seeds[0]),"Field",
					self.best_gmm["Synthetic_{0}".format(seeds[0])]["Field"])
				cmd_fld   = "cp {0} {1}".format(model_fld,dir_model)
				os.system(cmd_fld)
				#-----------------------------------------------------

				#-------- Best gmm same as seed_0 ---------------
				self.best_gmm[instance] = self.best_gmm[
								"Synthetic_{0}".format(seeds[0])]
				#-------------------------------------------------------
			#---------------------------------------------------------------

			#------------ Cluster model ------------------------------
			model_cls = self.file_model_base.format("Real","Cluster",
							self.best_gmm["Real"]["Cluster"])
			cmd_cls   = "cp {0} {1}".format(model_cls,dir_model)
			os.system(cmd_cls)
			#---------------------------------------------------------

			#----- Compute probabilities -----------------
			self.compute_probabilities(instance=instance)
			#---------------------------------------------

	def find_probability_threshold(self,seeds,bins=4,prob_steps=1000,
		covariate="g",metric="MCC"):
		#-------- Libraries -------------------
		from Quality import ClassifierQuality
		#--------------------------------------

		file_plot = self.file_qlt_base.format(covariate,metric,"pdf")
		file_tex  = self.file_qlt_base.format(covariate,metric,"tex")
		file_thr  = self.file_qlt_base.format(covariate,metric,"pkl")
		

		os.makedirs(self.path_main+"/Classification/",exist_ok=True)

		dfs = []
		for seed in seeds:
			#------------ File ----------------------------
			instance  = "Synthetic_{0}".format(seed)
			file_hdf  = self.file_hdf_base.format(instance)
			
			#------------- Reading -------------------------------
			print("Reading catalogue and probabilities ...")
			df = pd.read_hdf(file_hdf,key="catalogue",
					columns=["Cluster","prob_cls",covariate])
			#-----------------------------------------------------

			#-- Append ----
			dfs.append(df)
			#--------------

		print("Analyzing classifier quality ...")
		clq = ClassifierQuality(file_data=dfs,
								variate="prob_cls",
								covariate=covariate,
								true_class="Cluster")
		clq.confusion_matrix(bins=bins,
							prob_steps=prob_steps,
							metric=metric)
		clq.plots(file_plot=file_plot)
		clq.save(file_tex=file_tex)

		self.file_thresholds = file_thr

		

	def plot_members(self,probability_threshold=None,instance="Real"):

		file_hdf  = self.file_hdf_base.format(instance)

		#------------- Reading -------------------------------
		print("Reading catalogue ...")
		df_cat = pd.read_hdf(file_hdf,key="catalogue")
		#-----------------------------------------------------

		#----- Members ------------------------------
		df_mem  = df_cat.loc[df_cat["Member"]].copy()
		#--------------------------------------------

		#----- Candidates --------------------------------
		print("Selecting candidates ...")
		if isinstance(probability_threshold,float):
			mask_cnd = df_cat["prob_cls"] >= probability_threshold
			df_cnd   = df_cat.loc[mask_cnd].copy()
			del df_cat

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

				#----------- Temporal DF ---------
				tmp = df_cat.iloc[idx].copy()
				#----------------------------------

				#------------- Members ----------------------------
				tmp_mem = tmp[tmp["prob_cls"] >= threshold].copy()
				tmp_mem["Strategy"] = strategy
				#-------------------------------------------------

				#------------ Append -------------------------------------------
				dfs.append(tmp_mem)
				#---------------------------------------------------------------

			#--------- Concatenate and extract -----
			df = pd.concat(dfs)
			df_cnd = df[df["Strategy"] != "All"].copy()
			del df_cat
			#-------------------------------------------
		#-------------------------------------------------

		#---------- IDs ------------------------
		ids_cnd = df_cnd["source_id"].to_numpy()
		ids_mem = df_mem["source_id"].to_numpy()
		#---------------------------------------

		#-------- Summary----------------------------
		ids_common = np.intersect1d(ids_mem,ids_cnd)
		ids_reject = np.setdiff1d(ids_mem,ids_cnd)
		ids_new    = np.setdiff1d(ids_cnd,ids_mem)
		print("Common: {0}".format(len(ids_common)))
		print("Rejected: {0}".format(len(ids_reject)))
		print("New: {0}".format(len(ids_new)))
		#---------------------------------------------

		#----------- Color ---------------------------
		df_mem["g_rp"] = df_mem["g"] - df_mem["rp"]
		df_cnd["g_rp"] = df_cnd["g"] - df_cnd["rp"]
		#---------------------------------------------

		#----------- Absoulute magnitude -----------
		df_mem["G"] = df_mem["g"] + 5.*( 1.0 - 
						np.log10(1000./df_mem["parallax"]))
		df_cnd["G"] = df_cnd["g"] + 5.*( 1.0 - 
						np.log10(1000./df_cnd["parallax"]))
		#-------------------------------------------

		#--------------------------------------------------------
		pdf = PdfPages(filename=self.file_mem_plot)
		for i,plot in enumerate(self.plots):
			fig = plt.figure()
			ax  = plt.gca()
			#--------- Sources --------------------------
			ax.scatter(x=df_mem[plot[0]],y=df_mem[plot[1]],
						marker=".",s=5,
						c=df_mem["prob_cls"],
						vmin=0,vmax=1,
						cmap=self.cmap_prob,
						zorder=0,
						label="Members")
			scb = ax.scatter(x=df_cnd[plot[0]],y=df_cnd[plot[1]],
						marker="$\u25A1$",
						s=10,c=df_cnd["prob_cls"],
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

		#--------- Save candidates -------
		df_cnd.to_csv(self.file_mem_data,index=False)
		#=============================================================

	def run_real(self,file_catalogue,file_members,n_samples=100000,chunks=1):

		#-------------------- Synthetic --------------------------
		if not os.path.isfile(self.file_smp_base.format("Real")):
			self.generate_true_cluster(
						file_kalkayotl=self.file_kalkayotl,
						n_samples=n_samples)
		#---------------------------------------------------------

		#------------- Assemble -----------------------------------
		if not os.path.isfile(self.file_data_base.format("Real")):
			self.assemble_data(file_catalogue=file_catalogue,
								file_members=file_members,
								instance="Real")
		#------------------------------------------------------
		
		#--------------- Infer models ---------------------
		self.infer_models(case="Field",instance="Real")
		self.infer_models(case="Cluster",instance="Real")
		#-------------------------------------------------

		#------------- Select best models --------------------------
		if "Real" not in self.best_gmm:
			self.select_best_model(case="Field",instance="Real")
			self.select_best_model(case="Cluster",instance="Real")
			print("The best real GMM models are:")
			print(self.best_gmm)
		#-----------------------------------------------------------

		#----------------- Plot best models ---------------
		self.plot_model(case="Field",instance="Real")
		self.plot_model(case="Cluster",instance="Real")
		#--------------------------------------------------

		#-------- Probabilities ---------------------
		self.compute_probabilities(instance="Real",chunks=chunks)
		#--------------------------------------------

	def run_synthetic(self,seeds,file_catalogue,n_field=int(1e5)):
		#----------- Synthetic data --------------------------------
		self.generate_synthetic(file_catalogue=file_catalogue,
							   n_field=n_field,
							   seeds=seeds)
		self.compute_probabilities_synthetic(seeds)
		#----------------------------------------------------------



if __name__ == "__main__":
	#----------------- Directories ------------------------
	dir_repos= "/scratch/jolivares/Repos/"
	dir_main = "/scratch/jolivares/OCs/ComaBer/Mecayotl/"
	dir_cats = "/scratch/jolivares/OCs/ComaBer/Catalogues/"
	#-------------------------------------------------------

	#----------- Files --------------------------------------------
	file_kalkayotl = dir_main + "Real/Data/Cluster_statistics.csv"
	file_members   = dir_main + "Real/Data/members_Furnkranz+2019.csv"
	file_new_mem   = dir_main + "Real/Data/members_mecayotl.csv"
	file_rel_cat   = dir_cats + "ComaBer_33deg.fits"
	file_syn_cat   = dir_cats + "ComaBer_33deg_simulation.fits"
	#--------------------------------------------------------------

	#----- Miscellaneous ------------------------------------------------
	seeds = range(10)
	photometric_args = {
		"log_age": np.log10(8.0e8),    
		"metallicity":0.012,
		"Av": 0.0,         
		"mass_limits":[0.1,2.5], 
		"bands":["V","I","G","BP","RP"]
	}
	#---------------------------------------------------------------------
	

	mcy = Mecayotl(file_kalkayotl=file_kalkayotl,
				   photometric_args=photometric_args,
				   path_main=dir_main,
				   nc_cluster=range(1,11,1),
				   nc_field=range(1,15,1),
				   path_amasijo=dir_repos+"Amasijo/",
				   path_mcmichael=dir_repos+"McMichael/")

	mcy.run_real(file_catalogue=file_rel_cat,file_members=file_members)
	print(mcy.best_gmm)
	mcy.run_synthetic(seeds=seeds,file_catalogue=file_syn_cat,
					  n_field=int(1e6))
	
	mcy.find_probability_threshold(seeds=seeds,bins=5)
	mcy.plot_members(instance="Real")
	#----------------------------------------------------------------------------

	


