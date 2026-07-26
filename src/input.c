/*
 * input.c
 *   various input/output routines for gk package
 *
 *   copyright by Johannes Tausch
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "gkGlobal.h"
#include "gk.h"
#define MAXCOND 50
#define DENSITY 10    /* for mkMultiSpheres() */

int nSurf = 0;

/* Build file path: use panelfile as-is if it contains '/', else prepend fpath */
static void buildPath(char *fname, size_t fnameSize, const char *fpath, const char *panelfile, const char *ext) {
  int nwritten;
  if (strchr(panelfile, '/') != NULL) {
    nwritten = snprintf(fname, fnameSize, "%s%s", panelfile, ext);
  } else {
    nwritten = snprintf(fname, fnameSize, "%s%s%s", fpath, panelfile, ext);
  }

  if (nwritten < 0 || (size_t)nwritten >= fnameSize) {
    fprintf(stderr, "Error: generated path is too long for buffer\n");
    exit(1);
  }
}

static int parsePqrAtomLine(const char *line, double *x, double *y, double *z,
                            double *charge, double *radius, int *isMalformedAtomLine) {
  char copy[512];
  char *tokens[32];
  char *tok;
  int ntok = 0;
  char *endptr;

  *isMalformedAtomLine = 0;
  snprintf(copy, sizeof(copy), "%s", line);

  tok = strtok(copy, " \t\r\n");
  while (tok != NULL && ntok < (int)(sizeof(tokens) / sizeof(tokens[0]))) {
    tokens[ntok++] = tok;
    tok = strtok(NULL, " \t\r\n");
  }

  if (ntok <= 0) {
    return 0;
  }

  if (strcmp(tokens[0], "ATOM") != 0 && strcmp(tokens[0], "HETATM") != 0) {
    return 0;
  }

  if (ntok < 10) {
    *isMalformedAtomLine = 1;
    return 0;
  }

  *radius = strtod(tokens[ntok - 1], &endptr);
  if (*endptr != '\0') {
    *isMalformedAtomLine = 1;
    return 0;
  }
  *charge = strtod(tokens[ntok - 2], &endptr);
  if (*endptr != '\0') {
    *isMalformedAtomLine = 1;
    return 0;
  }
  *z = strtod(tokens[ntok - 3], &endptr);
  if (*endptr != '\0') {
    *isMalformedAtomLine = 1;
    return 0;
  }
  *y = strtod(tokens[ntok - 4], &endptr);
  if (*endptr != '\0') {
    *isMalformedAtomLine = 1;
    return 0;
  }
  *x = strtod(tokens[ntok - 5], &endptr);
  if (*endptr != '\0') {
    *isMalformedAtomLine = 1;
    return 0;
  }

  return 1;
}

static const char *getNanoShaperBin(void) {
  const char *bin = getenv("FABIPB_NANOSHAPER_BIN");
  if (bin != NULL && bin[0] != '\0') {
    return bin;
  }

#ifdef FMM_PB_DEFAULT_NANOSHAPER_BIN
  if (access(FMM_PB_DEFAULT_NANOSHAPER_BIN, X_OK) == 0) {
    return FMM_PB_DEFAULT_NANOSHAPER_BIN;
  }
#endif

  if (access("./nanoshaper-master/build/NanoShaper", X_OK) == 0) {
    return "./nanoshaper-master/build/NanoShaper";
  }

  if (access("../nanoshaper-master/build/NanoShaper", X_OK) == 0) {
    return "../nanoshaper-master/build/NanoShaper";
  }

  return "NanoShaper";
}

static void removePanelArtifacts(const char *fpath, const char *panelfile) {
  char fname[256];

  buildPath(fname, sizeof(fname), fpath, panelfile, ".xyzr");
  remove(fname);
  buildPath(fname, sizeof(fname), fpath, panelfile, ".vert");
  remove(fname);
  buildPath(fname, sizeof(fname), fpath, panelfile, ".face");
  remove(fname);
}

static void joinPath(char *out, size_t outSize, const char *dir, const char *name) {
  int nwritten = snprintf(out, outSize, "%s/%s", dir, name);
  if (nwritten < 0 || (size_t)nwritten >= outSize) {
    fprintf(stderr, "Error: generated path is too long for buffer\n");
    exit(1);
  }
}

static void makeAbsolutePath(char *out, size_t outSize, const char *path) {
  char cwd[512];
  int nwritten;

  if (path[0] == '/') {
    nwritten = snprintf(out, outSize, "%s", path);
  } else {
    if (getcwd(cwd, sizeof(cwd)) == NULL) {
      fprintf(stderr, "Error: getcwd failed while building absolute path\n");
      exit(1);
    }
    nwritten = snprintf(out, outSize, "%s/%s", cwd, path);
  }
  if (nwritten < 0 || (size_t)nwritten >= outSize) {
    fprintf(stderr, "Error: generated absolute path is too long for buffer\n");
    exit(1);
  }
}

static void shellQuote(char *out, size_t outSize, const char *in) {
  size_t pos = 0;
  size_t i;

  if (outSize < 3) {
    fprintf(stderr, "Error: shell quote buffer too small\n");
    exit(1);
  }
  out[pos++] = '\'';
  for (i = 0; in[i] != '\0'; i++) {
    if (in[i] == '\'') {
      const char *esc = "'\\''";
      size_t j;
      for (j = 0; esc[j] != '\0'; j++) {
        if (pos + 1 >= outSize) {
          fprintf(stderr, "Error: shell-quoted path is too long for buffer\n");
          exit(1);
        }
        out[pos++] = esc[j];
      }
    } else {
      if (pos + 1 >= outSize) {
        fprintf(stderr, "Error: shell-quoted path is too long for buffer\n");
        exit(1);
      }
      out[pos++] = in[i];
    }
  }
  if (pos + 1 >= outSize) {
    fprintf(stderr, "Error: shell-quoted path is too long for buffer\n");
    exit(1);
  }
  out[pos++] = '\'';
  out[pos] = '\0';
}

static void makeNanoTempDir(char *tmpDir, size_t tmpDirSize) {
  int attempt;
  long pid = (long)getpid();

  for (attempt = 0; attempt < 100; attempt++) {
    int nwritten = snprintf(tmpDir, tmpDirSize, ".fabipb_nanoshaper_%ld_%d", pid, attempt);
    if (nwritten < 0 || (size_t)nwritten >= tmpDirSize) {
      fprintf(stderr, "Error: NanoShaper temp directory path is too long\n");
      exit(1);
    }
    if (mkdir(tmpDir, 0700) == 0) {
      return;
    }
    if (errno != EEXIST) {
      fprintf(stderr, "Error: cannot create NanoShaper temp directory '%s'\n", tmpDir);
      exit(1);
    }
  }
  fprintf(stderr, "Error: could not create a unique NanoShaper temp directory\n");
  exit(1);
}

static void cleanupNanoTempDir(const char *tmpDir) {
  const char *files[] = {
    "surfaceConfiguration.prm",
    "nsout.txt",
    "stderror.txt",
    "triangleAreas.txt",
    "exposed.xyz",
    "exposedIndices.txt",
    "triangulatedSurf.face",
    "triangulatedSurf.vert"
  };
  char path[512];
  size_t i;

  for (i = 0; i < sizeof(files) / sizeof(files[0]); i++) {
    joinPath(path, sizeof(path), tmpDir, files[i]);
    remove(path);
  }
  rmdir(tmpDir);
}

/*
 * calculate area and normal
 */
double triangle_area(double v[3][3]){
  int i;
  double a[3], b[3], c[3], aa, bb, cc, ss, t_area;
  for (i=0;i<3;i++){
    a[i] = v[0][i]-v[1][i];
    b[i] = v[0][i]-v[2][i];
    c[i] = v[1][i]-v[2][i];
  }
  aa = sqrt(SQR(a[0])+SQR(a[1])+SQR(a[2]));
  bb = sqrt(SQR(b[0])+SQR(b[1])+SQR(b[2]));
  cc = sqrt(SQR(c[0])+SQR(c[1])+SQR(c[2]));
  ss = 0.5*(aa+bb+cc);
  t_area = sqrt(ss*(ss-aa)*(ss-bb)*(ss-cc));
  return(t_area);
}

/*
 * loadpanel returns a list of panel structs derived from passed data:
 * shape, vertices, and type.
 */
panel *loadPanel(char *panelfile, const char *meshParam, int *numSing, ssystem *sys,
                 const char *meshControlName, double meshControlValue,
                 const char *backendParamName, double backendParamValue) {
  int i, j, k, ii, shape, type, nSurf, mesh_flag=sys->mesh_flag;
  panel *pnlList, *pnl;
  char fpath[256], fname[256];
  FILE *fp, *wfp;

  char c, line[512];
  double a1,a2,a3,b1,b2,b3;//a_norm,r0_norm,v0_norm;
  int i1,i2,i3,j1,j2,j3,ierr,iface[3],jface[3],ialert;
  double den,prob_rds,xx[3],yy[3],face[3][3],tface[3][3],s_area;
  double **sptpos, **sptnrm;
  int **extr_v, **extr_f, *nvert;
  double dist_local, area_local, cpuf;
  int nspt, natm, nface;
  int malformedAtomLines, zeroRadiusAtoms, parsedAtom, filledAtoms, meshAtoms;

  double *nrm, len;

  /* read in vertices */
  sys->nChar = 0;
  meshAtoms = 0;
  sprintf(fpath,"test_proteins/");
  buildPath(fname, sizeof(fname), fpath, panelfile, ".pqr");
  fp=fopen(fname,"r");
  if (fp == NULL) {
    fprintf(stderr, "Error: cannot open PQR file '%s'\n", fname);
    fprintf(stderr, "Usage: fabipb <pdb_id>  (e.g. fabipb 1a63)\n");
    fprintf(stderr, "       or fabipb <path> (e.g. fabipb test_proteins/1a63)\n");
    exit(1);
  }
  buildPath(fname, sizeof(fname), fpath, panelfile, ".xyzr");
  wfp=fopen(fname,"w");
  malformedAtomLines = 0;
  zeroRadiusAtoms = 0;
  while (fgets(line, sizeof(line), fp) != NULL) {
    int malformedThisLine = 0;
    parsedAtom = parsePqrAtomLine(line, &a1, &a2, &a3, &b1, &b2, &malformedThisLine);
    if (malformedThisLine) {
      malformedAtomLines++;
      continue;
    }
    if (!parsedAtom) {
      continue;
    }
    if (b2 <= 0.0) {
      zeroRadiusAtoms++;
    } else {
      fprintf(wfp,"%f %f %f %f\n",a1,a2,a3,b2);
      meshAtoms++;
    }
    sys->nChar++;
  }
  if (zeroRadiusAtoms > 0) {
    fprintf(stderr, "Info: kept %d zero-radius atoms as charges and excluded them from mesh for '%s'\n",
            zeroRadiusAtoms, panelfile);
  }
  if (malformedAtomLines > 0) {
    fprintf(stderr, "Warning: skipped %d malformed ATOM/HETATM lines from '%s'\n", malformedAtomLines, panelfile);
  }
  printf("Mesh input: panel=%s mesh_atoms=%d charge_atoms=%d mode=%s param=%s\n",
         panelfile,
         meshAtoms,
         sys->nChar,
         (mesh_flag == 1) ? "msms" : ((mesh_flag == 2) ? "nanoshaper" : "unknown"),
         meshParam);
  printf("Mesh control: %s=%g resolved-%s=%g\n",
         meshControlName, meshControlValue, backendParamName, backendParamValue);
  fclose(fp);
  fclose(wfp);

  if (meshAtoms <= 0) {
    fprintf(stderr, "Error: no positive-radius ATOM/HETATM records were parsed from '%s'\n", panelfile);
    exit(1);
  }
  if (sys->nChar <= 0) {
    fprintf(stderr, "Error: no ATOM/HETATM charge records were parsed from '%s'\n", panelfile);
    exit(1);
  }

  CALLOC(sys->pos, 3*sys->nChar, double);
  CALLOC(sys->chr, sys->nChar, double);
  buildPath(fname, sizeof(fname), fpath, panelfile, ".pqr");
  fp=fopen(fname,"r");
  filledAtoms = 0;
  while (fgets(line, sizeof(line), fp) != NULL && filledAtoms < sys->nChar) {
    int malformedThisLine = 0;
    parsedAtom = parsePqrAtomLine(line, &a1, &a2, &a3, &b1, &b2, &malformedThisLine);
    if (!parsedAtom || malformedThisLine) {
      continue;
    }
    sys->pos[3*filledAtoms]=a1;
    sys->pos[3*filledAtoms+1]=a2;
    sys->pos[3*filledAtoms+2]=a3;
    sys->chr[filledAtoms]=b1;
    filledAtoms++;
  }
  fclose(fp);
  if (filledAtoms != sys->nChar) {
    fprintf(stderr, "Error: parsed atom count mismatch for '%s' (%d expected, %d loaded)\n",
            panelfile, sys->nChar, filledAtoms);
    exit(1);
  }

  if ( mesh_flag == 1 ) {
  /* run msms */
    {
      char xyzrpath[256], ofbase[256];
      int nwritten;
      buildPath(xyzrpath, sizeof(xyzrpath), fpath, panelfile, ".xyzr");
      buildPath(ofbase, sizeof(ofbase), fpath, panelfile, "");
      nwritten = snprintf(fname, sizeof(fname),
                          "msms -if %s -prob 1.4 -de %s -of %s > msms.output",
                          xyzrpath, meshParam, ofbase);
      if (nwritten < 0 || (size_t)nwritten >= sizeof(fname)) {
        fprintf(stderr, "Error: msms command is too long for buffer\n");
        exit(1);
      }
    }
    //printf("%s\n",fname);
    ierr = system(fname);
    if (ierr != 0) {
      fprintf(stderr, "Error: msms failed while processing '%s'\n", panelfile);
      removePanelArtifacts(fpath, panelfile);
      exit(1);
    }
  } else if ( mesh_flag == 2 ) {
    const char *nanoBin = getNanoShaperBin();
    int nwritten;
    char tmpDir[256], prmPath[512], faceTmp[512], vertTmp[512];
    char xyzrpath[256], xyzrAbs[512], nanoBinPath[512];
    char quotedTmpDir[768], quotedNanoBin[768], command[2048];

    makeNanoTempDir(tmpDir, sizeof(tmpDir));
    joinPath(prmPath, sizeof(prmPath), tmpDir, "surfaceConfiguration.prm");
    joinPath(faceTmp, sizeof(faceTmp), tmpDir, "triangulatedSurf.face");
    joinPath(vertTmp, sizeof(vertTmp), tmpDir, "triangulatedSurf.vert");
    buildPath(xyzrpath, sizeof(xyzrpath), fpath, panelfile, ".xyzr");
    makeAbsolutePath(xyzrAbs, sizeof(xyzrAbs), xyzrpath);
    if (strchr(nanoBin, '/') != NULL) {
      makeAbsolutePath(nanoBinPath, sizeof(nanoBinPath), nanoBin);
    } else {
      nwritten = snprintf(nanoBinPath, sizeof(nanoBinPath), "%s", nanoBin);
      if (nwritten < 0 || (size_t)nwritten >= sizeof(nanoBinPath)) {
        fprintf(stderr, "Error: NanoShaper executable path is too long\n");
        cleanupNanoTempDir(tmpDir);
        exit(1);
      }
    }

    wfp = fopen(prmPath, "w");
    if (wfp == NULL) {
      fprintf(stderr, "Error: cannot write NanoShaper configuration '%s'\n", prmPath);
      cleanupNanoTempDir(tmpDir);
      exit(1);
    }
    fprintf(wfp, "Grid_scale = %s\n", meshParam);
    fprintf(wfp, "Grid_perfil = 90.0\n");
    fprintf(wfp, "XYZR_FileName = %s\n", xyzrAbs);
    fprintf(wfp, "Build_epsilon_maps = false\n");
    fprintf(wfp, "Build_status_map = false\n");
    fprintf(wfp, "Save_Mesh_MSMS_Format = true\n");
    fprintf(wfp, "Compute_Vertex_Normals = true\n");
    fprintf(wfp, "Surface = ses\n");

    fprintf(wfp, "Smooth_Mesh = true\n");
    fprintf(wfp, "Skin_Surface_Parameter = %f\n", 0.45);

    fprintf(wfp, "Cavity_Detection_Filling = false\n");
    fprintf(wfp, "Conditional_Volume_Filling_Value = %f\n", 11.4);
    fprintf(wfp, "Keep_Water_Shaped_Cavities = false\n");
    fprintf(wfp, "Probe_Radius = %f\n", 1.4);
    fprintf(wfp, "Accurate_Triangulation = true\n");
    fprintf(wfp, "Triangulation = true\n");
    fprintf(wfp, "Check_duplicated_vertices = true\n");
    fprintf(wfp, "Save_Status_map = false\n");
    fprintf(wfp, "Save_PovRay = false\n");
    fclose(wfp);

    shellQuote(quotedTmpDir, sizeof(quotedTmpDir), tmpDir);
    shellQuote(quotedNanoBin, sizeof(quotedNanoBin), nanoBinPath);
    nwritten = snprintf(command, sizeof(command), "cd %s && %s surfaceConfiguration.prm >> nsout.txt",
                        quotedTmpDir, quotedNanoBin);
    if (nwritten < 0 || (size_t)nwritten >= sizeof(command)) {
      fprintf(stderr, "Error: NanoShaper command is too long for buffer\n");
      cleanupNanoTempDir(tmpDir);
      exit(1);
    }
    ierr = system(command);
    if (ierr != 0) {
      fprintf(stderr, "Error: NanoShaper failed while processing '%s' using executable '%s'\n",
              panelfile, nanoBin);
      cleanupNanoTempDir(tmpDir);
      removePanelArtifacts(fpath, panelfile);
      exit(1);
    }

    buildPath(fname, sizeof(fname), fpath, panelfile, ".face");
    if (rename(faceTmp, fname) != 0) {
      fprintf(stderr, "Error: cannot move NanoShaper face output to '%s'\n", fname);
      cleanupNanoTempDir(tmpDir);
      removePanelArtifacts(fpath, panelfile);
      exit(1);
    }
    buildPath(fname, sizeof(fname), fpath, panelfile, ".vert");
    if (rename(vertTmp, fname) != 0) {
      fprintf(stderr, "Error: cannot move NanoShaper vertex output to '%s'\n", fname);
      cleanupNanoTempDir(tmpDir);
      removePanelArtifacts(fpath, panelfile);
      exit(1);
    }

    cleanupNanoTempDir(tmpDir);
  } else {
    fprintf(stderr, "Error: unsupported mesh mode %d (use 1 for MSMS or 2 for NanoShaper)\n",
            mesh_flag);
    removePanelArtifacts(fpath, panelfile);
    exit(1);
  }
  /*======================================================================*/

  /* read in vert */
  buildPath(fname, sizeof(fname), fpath, panelfile, ".vert");
  fp=fopen(fname,"r");
  if (fp == NULL) {
    fprintf(stderr, "Error: cannot open vertices file '%s' (generate the mesh with -m=1 or -m=2 first)\n", fname);
    exit(1);
  }

  /* open the file and read through the first two rows */
  for ( i=1; i<=2; i++ ) {
    while ( (c=getc(fp))!='\n' ){
   }
  }

  if ( mesh_flag != 2 ) {
    ierr=fscanf(fp,"%d %d %lf %lf ",&nspt,&natm,&den,&prob_rds);
    //printf("nspt=%d, natm=%d, den=%lf, prob=%lf\n", nspt,natm,den,prob_rds);
  } else if ( mesh_flag == 2 ) {
    ierr = fscanf(fp, "%d", &nspt);
  }

  /*allocate variables for vertices file*/
  CALLOC(sptpos, 3, double*);
  CALLOC(sptnrm, 3, double*);
  for ( i=0; i<3; i++ ) {
    CALLOC(sptpos[i], nspt, double);
    CALLOC(sptnrm[i], nspt, double);
  }

  for ( i=0; i<=nspt-1; i++ ) {
    ierr=fscanf(fp,"%lf %lf %lf %lf %lf %lf %d %d %d",&a1,&a2,&a3,&b1,&b2,&b3,&i1,&i2,&i3);

    sptpos[0][i]=a1;
    sptpos[1][i]=a2;
    sptpos[2][i]=a3;
    sptnrm[0][i]=b1;
    sptnrm[1][i]=b2;
    sptnrm[2][i]=b3;
  }
  fclose(fp);

  /* read in faces */
  buildPath(fname, sizeof(fname), fpath, panelfile, ".face");
  fp=fopen(fname,"r");
  if (fp == NULL) {
    fprintf(stderr, "Error: cannot open faces file '%s'\n", fname);
    exit(1);
  }
  for ( i=1; i<=2; i++ ) { while ((c=getc(fp))!='\n'){} }

  if ( mesh_flag != 2 ) {
    ierr=fscanf(fp,"%d %d %lf %lf ",&nface,&natm,&den,&prob_rds);
  //printf("nface=%d, natm=%d, den=%lf, prob=%lf\n", nface,natm,den,prob_rds);
  } else if ( mesh_flag == 2 ) {
    ierr = fscanf(fp, "%d", &nface);
  }

  printf("Mesh raw counts: vertices=%d faces=%d\n", nspt, nface);

  /* allocate variables for vertices file */
  CALLOC(nvert, 3*nface, int);

  for ( i=0; i<=nface-1; i++ ) {
    ierr=fscanf(fp,"%d %d %d %d %d",&j1,&j2,&j3,&i1,&i2);
    nvert[3*i]=j1;
    nvert[3*i+1]=j2;
    nvert[3*i+2]=j3;
  }
  fclose(fp);

  /* we delete ill performence triangles */
  s_area = 0.0;
  if (sys->benchmarkMode > 0) {
    printf("#ele=%d, ",nface);
  }
  //printf("Number of vertices = %d\n",nspt);

  pnlList = NULL;
  *numSing = 0;
  for ( i=0; i<nface; i++ ) {

    for ( j=0; j<3; j++ ) {
      iface[j]=nvert[3*i+j];
      xx[j]=0.0;
    }
    for ( j=0; j<3; j++ ) {
      for ( k=0; k<3; k++ ) {
        face[k][j]=sptpos[j][iface[k]-1];
	      xx[j] += 1.0/3.0*(face[k][j]);
      }
    }
    /* compute the area of each triangule */
    area_local = triangle_area(face);
    /* if the point is too close with 10 points infront, delete it */
    ialert=0;
    for ( j=i-10; (j>=0&&j<i); j++ ) { /* like k=max(1,i-10), i-1 in fortran */
      for ( k=0; k<3; k++ ) {
        jface[k] = nvert[3*j+k];
        yy[k]=0.0;
      }
      for ( k=0; k<3; k++ ) {
        for ( ii=0; ii<3; ii++ ) {
          tface[ii][k] = sptpos[k][jface[ii]-1];
          yy[k] += 1.0/3.0*(tface[ii][k]);
        }
      }
      dist_local = 0.0;
      for( k=0; k<3; k++ ) dist_local+=(xx[k]-yy[k])*(xx[k]-yy[k]);/* dot_product */
      //dist_local=sqrt(dist_local);
      if ( dist_local<1.0e-5 ) {
        ialert=1;
        //printf("particles %d and %d are too close: %e\n", i,j,dist_local);
      }
    }

    /* allocate and fill the panel */
    if ( area_local>=1.0e-5 && ialert == 0 ) {
      (*numSing)++;
      if ( pnlList == NULL ) {
        CALLOC(pnlList, 1, panel);
        pnl = pnlList;
      }
      else{
        CALLOC(pnl->next, 1, panel);
        pnl = pnl->next;
      }
  /* Fill in corners. */
      for ( j=0; j<3; j++ ) {
        for ( k=0; k<3; k++ ) {
          pnl->vtx[j][k] = sptpos[k][iface[j]-1];
          pnl->nrm[j][k] = sptnrm[k][iface[j]-1];
        }
      }

      for ( j=0; j<3; j++ ) {
        pnl->a[0][j] = pnl->vtx[2][j] - pnl->vtx[1][j];
        pnl->a[1][j] = pnl->vtx[0][j] - pnl->vtx[2][j];
        pnl->a[2][j] = pnl->vtx[1][j] - pnl->vtx[0][j];
      }

      nrm = pnl->normal;
      nrm[0] = pnl->nrm[0][0]/2. + (pnl->nrm[1][0] + pnl->nrm[2][0])/4.;
      nrm[1] = pnl->nrm[0][1]/2. + (pnl->nrm[1][1] + pnl->nrm[2][1])/4.;
      nrm[2] = pnl->nrm[0][2]/2. + (pnl->nrm[1][2] + pnl->nrm[2][2])/4.;
      len = sqrt(SQR(nrm[0]) + SQR(nrm[1]) + SQR(nrm[2]));
      for ( j=0; j<3; j++ ) nrm[j] /= len;

      pnl->shape = 3;
      pnl->nSurf = *numSing;
      pnl->area = area_local;
      s_area += area_local;
    }
  }

  printf("Mesh filtered panels: kept=%d area=%f\n", *numSing, s_area);
  //printf("%d ugly faces are deleted\n", nface-*numSing);

  removePanelArtifacts(fpath, panelfile);

  for (i=0;i<3;i++){
    free(sptpos[i]);
    free(sptnrm[i]);
  }
  free(sptpos);
  free(sptnrm);
  free(nvert);

  return pnlList;
} /* loadPanel */
