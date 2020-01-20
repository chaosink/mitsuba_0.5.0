/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob and others.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <mitsuba/render/scene.h>
#include <mitsuba/core/statistics.h>
#include <mitsuba/render/renderproc.h>
#include <mitsuba/core/plugin.h>

#include <iomanip>

MTS_NAMESPACE_BEGIN

template<typename T>
inline std::unique_ptr<T[]> zeroAlloc(size_t size)
{
	std::unique_ptr<T[]> result(new T[size]);
	std::memset(result.get(), 0, size*sizeof(T));
	return std::move(result);
}

class OutputBuffer
{
	Vector2i _res;

	std::unique_ptr<Vector[]> _buffer;
	std::unique_ptr<float[]> _variance;
	std::unique_ptr<uint[]> _sampleCount;

public:
	OutputBuffer(Vector2i res)
		: _res(res)
	{
		size_t numPixels = res.x * res.y;

		_buffer = zeroAlloc<Vector>(numPixels);
		_variance = zeroAlloc<float>(numPixels);
		_sampleCount = zeroAlloc<uint>(numPixels);
	}

	void addSample(Point2i pixel, Vector c)
	{
		if (std::isnan(c.x) || std::isnan(c.y) || std::isnan(c.z))
			return;

		int idx = pixel.x + pixel.y*_res.x;
		uint sampleIdx = _sampleCount[idx]++;
		if (_variance) {
			Vector curr = _buffer[idx];
			Vector delta = c - curr;
			curr += delta/(sampleIdx + 1);
			_variance[idx] += dot(delta, (c - curr)) / 3.f;
		}

		_buffer[idx] += (c - _buffer[idx])/(sampleIdx + 1);
	}

	inline Vector operator[](uint idx) const
	{
		return _buffer[idx];
	}

	inline Vector get(int x, int y) const
	{
		return _buffer[x + y*_res.x];
	}

	inline float variance(int x, int y) const
	{
		return _variance[x + y*_res.x]/std::max(uint(1), _sampleCount[x + y*_res.x] - 1);
	}

	inline uint sampleCount(int x, int y) const
	{
		return _sampleCount[x + y*_res.x];
	}

	// void WriteEXR(std::string &avg_name, std::string &var_name) {
	// 	ref<Bitmap> avg_img = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat16, _res);
	// 	ref<Bitmap> var_img = new Bitmap(Bitmap::ELuminance, Bitmap::EFloat16, _res);
	// 	for(int y = 0; y < _res.y; y++)
	// 		for(int x = 0; x < _res.x; x++) {
	// 			Spectrum avg;
	// 			Vector rgb = get(x, y);
	// 			avg[0] = rgb.x;
	// 			avg[1] = rgb.y;
	// 			avg[2] = rgb.z;
	// 			avg_img->setPixel(Point2i(x, y), avg);
	//
	// 			Spectrum var;
	// 			var[0] = variance(x, y) / sampleCount(x, y);
	// 			var_img->setPixel(Point2i(x, y), var);
	// 		}
	// 	avg_img->write(Bitmap::EOpenEXR, avg_name);
	// 	var_img->write(Bitmap::EOpenEXR, var_name);
	// }
};

static StatsCounter avgPathLength("Path tracer", "Average path length", EAverage);

/*! \plugin{path}{Path tracer}
 * \order{2}
 * \parameters{
 *     \parameter{maxDepth}{\Integer}{Specifies the longest path depth
 *         in the generated output image (where \code{-1} corresponds to $\infty$).
 *	       A value of \code{1} will only render directly visible light sources.
 *	       \code{2} will lead to single-bounce (direct-only) illumination,
 *	       and so on. \default{\code{-1}}
 *	   }
 *	   \parameter{rrDepth}{\Integer}{Specifies the minimum path depth, after
 *	      which the implementation will start to use the ``russian roulette''
 *	      path termination criterion. \default{\code{5}}
 *	   }
 *     \parameter{strictNormals}{\Boolean}{Be strict about potential
 *        inconsistencies involving shading normals? See the description below
 *        for details.\default{no, i.e. \code{false}}
 *     }
 *     \parameter{hideEmitters}{\Boolean}{Hide directly visible emitters?
 *        See page~\pageref{sec:hideemitters} for details.
 *        \default{no, i.e. \code{false}}
 *     }
 * }
 *
 * This integrator implements a basic path tracer and is a \emph{good default choice}
 * when there is no strong reason to prefer another method.
 *
 * To use the path tracer appropriately, it is instructive to know roughly how
 * it works: its main operation is to trace many light paths using \emph{random walks}
 * starting from the sensor. A single random walk is shown below, which entails
 * casting a ray associated with a pixel in the output image and searching for
 * the first visible intersection. A new direction is then chosen at the intersection,
 * and the ray-casting step repeats over and over again (until one of several
 * stopping criteria applies).
 * \begin{center}
 * \includegraphics[width=.7\textwidth]{images/integrator_path_figure.pdf}
 * \end{center}
 * At every intersection, the path tracer tries to create a connection to
 * the light source in an attempt to find a \emph{complete} path along which
 * light can flow from the emitter to the sensor. This of course only works
 * when there is no occluding object between the intersection and the emitter.
 *
 * This directly translates into a category of scenes where
 * a path tracer can be expected to produce reasonable results: this is the case
 * when the emitters are easily ``accessible'' by the contents of the scene. For instance,
 * an interior scene that is lit by an area light will be considerably harder
 * to render when this area light is inside a glass enclosure (which
 * effectively counts as an occluder).
 *
 * Like the \pluginref{direct} plugin, the path tracer internally relies on multiple importance
 * sampling to combine BSDF and emitter samples. The main difference in comparison
 * to the former plugin is that it considers light paths of arbitrary length to compute
 * both direct and indirect illumination.
 *
 * For good results, combine the path tracer with one of the
 * low-discrepancy sample generators (i.e. \pluginref{ldsampler},
 * \pluginref{halton}, or \pluginref{sobol}).
 *
 * \paragraph{Strict normals:}\label{sec:strictnormals}
 * Triangle meshes often rely on interpolated shading normals
 * to suppress the inherently faceted appearance of the underlying geometry. These
 * ``fake'' normals are not without problems, however. They can lead to paradoxical
 * situations where a light ray impinges on an object from a direction that is classified as ``outside''
 * according to the shading normal, and ``inside'' according to the true geometric normal.
 *
 * The \code{strictNormals}
 * parameter specifies the intended behavior when such cases arise. The default (\code{false}, i.e. ``carry on'')
 * gives precedence to information given by the shading normal and considers such light paths to be valid.
 * This can theoretically cause light ``leaks'' through boundaries, but it is not much of a problem in practice.
 *
 * When set to \code{true}, the path tracer detects inconsistencies and ignores these paths. When objects
 * are poorly tesselated, this latter option may cause them to lose a significant amount of the incident
 * radiation (or, in other words, they will look dark).
 *
 * The bidirectional integrators in Mitsuba (\pluginref{bdpt}, \pluginref{pssmlt}, \pluginref{mlt} ...)
 * implicitly have \code{strictNormals} set to \code{true}. Hence, another use of this parameter
 * is to match renderings created by these methods.
 *
 * \remarks{
 *    \item This integrator does not handle participating media
 *    \item This integrator has poor convergence properties when rendering
 *    caustics and similar effects. In this case, \pluginref{bdpt} or
 *    one of the photon mappers may be preferable.
 * }
 */
class MIPathTracer : public MonteCarloIntegrator {
public:
	MIPathTracer(const Properties &props)
		: MonteCarloIntegrator(props) {
		m_lightFieldResolution = props.getInteger("lightFieldResolution", 4);
		m_lightFieldBlockCount = m_lightFieldResolution * m_lightFieldResolution;
	}

	/// Unserialize from a binary data stream
	MIPathTracer(Stream *stream, InstanceManager *manager)
		: MonteCarloIntegrator(stream, manager) { }

	bool render(Scene *scene,
		RenderQueue *queue, const RenderJob *job,
		int sceneResID, int sensorResID, int samplerResID) {
		ref<Scheduler> sched = Scheduler::getInstance();
		ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
		ref<Film> film = sensor->getFilm();

		auto size = film->getSize();
		auto file_dir = scene->getDestinationFile().parent_path();

		// features buffer
		m_albedo = std::make_unique<OutputBuffer>(size);
		m_normal = std::make_unique<OutputBuffer>(size);
		m_depth = std::make_unique<OutputBuffer>(size);
		m_diffuse = std::make_unique<OutputBuffer>(size);
		m_specular = std::make_unique<OutputBuffer>(size);

		// light fields buffer
		m_lightFieldBlocks.resize(m_lightFieldBlockCount);
		for (auto &l: m_lightFieldBlocks)
			l = std::make_unique<OutputBuffer>(size);

		size_t nCores = sched->getCoreCount();
		const Sampler *sampler = static_cast<const Sampler *>(sched->getResource(samplerResID, 0));
		size_t sampleCount = sampler->getSampleCount();

		Log(EInfo, "Starting render job (%ix%i, " SIZE_T_FMT " %s, " SIZE_T_FMT
			" %s, " SSE_STR ") ..", film->getCropSize().x, film->getCropSize().y,
			sampleCount, sampleCount == 1 ? "sample" : "samples", nCores,
			nCores == 1 ? "core" : "cores");

		/* This is a sampling-based integrator - parallelize */
		ref<ParallelProcess> proc = new BlockedRenderProcess(job,
			queue, scene->getBlockSize());
		int integratorResID = sched->registerResource(this);
		proc->bindResource("integrator", integratorResID);
		proc->bindResource("scene", sceneResID);
		proc->bindResource("sensor", sensorResID);
		proc->bindResource("sampler", samplerResID);
		scene->bindUsedResources(proc);
		bindUsedResources(proc);
		sched->schedule(proc);

		m_process = proc;
		sched->wait(proc);
		m_process = NULL;
		sched->unregisterResource(integratorResID);

		{
			ref<ImageBlock> block = new ImageBlock(Bitmap::EMultiSpectrumAlphaWeight, size, film->getReconstructionFilter(), (int) (SPECTRUM_SAMPLES * 10 + 2), false);
			for (int x = 0; x < size.x; ++x)
				for (int y = 0; y < size.y; ++y) {
					float temp[3 * 10 + 2];
					{
						Vector v = m_albedo->get(x, y);
						temp[0] = v.x; temp[1] = v.y; temp[2] = v.z;
						float f = m_albedo->variance(x, y) / m_albedo->sampleCount(x, y);
						temp[3] = temp[4] = temp[5] = f;
					}
					{
						Vector v = m_normal->get(x, y);
						temp[6] = v.x; temp[7] = v.y; temp[8] = v.z;
						float f = m_normal->variance(x, y) / m_normal->sampleCount(x, y);
						temp[9] = temp[10] = temp[11] = f;
					}
					{
						Vector v = m_depth->get(x, y);
						temp[12] = v.x; temp[13] = v.y; temp[14] = v.z;
						float f = m_depth->variance(x, y) / m_depth->sampleCount(x, y);
						temp[15] = temp[16] = temp[17] = f;
					}
					{
						Vector v = m_diffuse->get(x, y);
						temp[18] = v.x; temp[19] = v.y; temp[20] = v.z;
						float f = m_diffuse->variance(x, y) / m_diffuse->sampleCount(x, y);
						temp[21] = temp[22] = temp[23] = f;
					}
					{
						Vector v = m_specular->get(x, y);
						temp[24] = v.x; temp[25] = v.y; temp[26] = v.z;
						float f = m_specular->variance(x, y) / m_specular->sampleCount(x, y);
						temp[27] = temp[28] = temp[29] = f;
					}
					temp[30] = 1.f; // alpha
					temp[31] = 1.f; // reconstruction filter weight
					Point2 pos = Point2(x + 0.5f, y + 0.5f);
					block->put(pos, temp);
				}

			auto props = Properties("hdrfilm");
			props.setInteger("width", size.x);
			props.setInteger("height", size.y);
			props.setBoolean("banner", false);
			props.setBoolean("attachLog", false);
			props.setString("pixelFormat", std::string("rgb,luminance") +
				",rgb,luminance" +
				",luminance,luminance" +
				",rgb,luminance" +
				",rgb,luminance");
			props.setString("channelNames", std::string("albedo,albedoVariance") +
				",normal,normalVariance" +
				",depth,depthVariance" +
				",diffuse,diffuseVariance" +
				",specular,specularVariance");
			ref<Film> featuresFilm = static_cast<Film*>(PluginManager::getInstance()->createObject(MTS_CLASS(Film), props));
			featuresFilm->clear();
			featuresFilm->put(block);
			auto file_name = scene->getDestinationFile().stem().string() + "_features.exr";
			featuresFilm->setDestinationFile(file_dir / file_name, 0);
			featuresFilm->develop(scene, queue->getRenderTime(job));
		}

		int numberWidth;
		{
			std::ostringstream oss;
			oss << m_lightFieldBlockCount;
			numberWidth = oss.str().size();
		}

#define MULTIPLE_EXRS_WITH_SINGLE_CHANNEL
#ifdef MULTIPLE_EXRS_WITH_SINGLE_CHANNEL // multiple EXRs with single channel
		ref<ImageBlock> block = new ImageBlock(Bitmap::ESpectrumAlphaWeight, size, film->getReconstructionFilter(), (int) (SPECTRUM_SAMPLES + 2), false);

		auto props = Properties("hdrfilm");
		props.setInteger("width", size.x);
		props.setInteger("height", size.y);
		props.setBoolean("banner", false);
		props.setBoolean("attachLog", false);
		props.setString("pixelFormat", "rgb");
		ref<Film> lightFieldFilm = static_cast<Film*>(PluginManager::getInstance()->createObject(MTS_CLASS(Film), props));

		for (int i = 0; i < m_lightFieldBlockCount; ++i) {
			block->clear();
			for (int x = 0; x < size.x; ++x)
				for (int y = 0; y < size.y; ++y) {
					Vector v = m_lightFieldBlocks[i]->get(x, y);
					float temp[3 + 2] = {
						v.x, v.y, v.z,
						1.f, // alpha
						1.f, // reconstruction filter weight
					};
					Point2 pos = Point2(x + 0.5f, y + 0.5f);
					block->put(pos, temp);
				}
			std::ostringstream oss;
			oss << "_light-field-" << std::setw(numberWidth) << std::setfill('0') << i << ".exr";
			lightFieldFilm->clear();
			lightFieldFilm->put(block);
			auto file_name = scene->getDestinationFile().stem().string() + oss.str();
			lightFieldFilm->setDestinationFile(file_dir / file_name, 0);
			lightFieldFilm->develop(scene, queue->getRenderTime(job));
		}
#else // single EXR with multiple channels
		ref<ImageBlock> block = new ImageBlock(Bitmap::EMultiSpectrumAlphaWeight, size, film->getReconstructionFilter(), (int) (SPECTRUM_SAMPLES * m_lightFieldBlockCount + 2), false);
		block->clear();
		for (int x = 0; x < size.x; ++x)
			for (int y = 0; y < size.y; ++y) {
				float temp[3 * m_lightFieldBlockCount + 2];
				for (int i = 0; i < m_lightFieldBlockCount; ++i) {
					Vector v = m_lightFieldBlocks[i]->get(x, y);
					temp[i * 3 + 0] = v.x;
					temp[i * 3 + 1] = v.y;
					temp[i * 3 + 2] = v.z;
				}
				temp[3 * m_lightFieldBlockCount + 0] = 1.f; // alpha
				temp[3 * m_lightFieldBlockCount + 1] = 1.f; // reconstruction filter weight
				Point2 pos = Point2(x + 0.5f, y + 0.5f);
				block->put(pos, temp);
			}

		auto props = Properties("hdrfilm");
		props.setInteger("width", size.x);
		props.setInteger("height", size.y);
		props.setBoolean("banner", false);
		props.setBoolean("attachLog", false);
		std::string pixelFormatStr("rgb");
		for (int i = 1; i < m_lightFieldBlockCount; ++i)
			pixelFormatStr += ",rgb";
		props.setString("pixelFormat", pixelFormatStr);
		std::string channelNamesStr;
		{
			std::ostringstream oss;
			oss << std::setw(numberWidth) << std::setfill('0') << 0;
			for (int i = 1; i < m_lightFieldBlockCount; ++i)
				oss << "," << std::setw(numberWidth) << std::setfill('0') << i;
			channelNamesStr = oss.str();
		}
		props.setString("channelNames", channelNamesStr);
		ref<Film> lightFieldFilm = static_cast<Film*>(PluginManager::getInstance()->createObject(MTS_CLASS(Film), props));
		lightFieldFilm->clear();
		lightFieldFilm->put(block);
		auto file_name = scene->getDestinationFile().stem().string() + "_light-field.exr";
		lightFieldFilm->setDestinationFile(file_dir / file_name, 0);
		lightFieldFilm->develop(scene, queue->getRenderTime(job));
#endif
		return proc->getReturnStatus() == ParallelProcess::ESuccess;
	}

	void renderBlock(const Scene *scene,
		const Sensor *sensor, Sampler *sampler, ImageBlock *block,
		const bool &stop, const std::vector< TPoint2<uint8_t> > &points) const {

		Float diffScaleFactor = 1.0f /
			std::sqrt((Float) sampler->getSampleCount());

		bool needsApertureSample = sensor->needsApertureSample();
		bool needsTimeSample = sensor->needsTimeSample();

		RadianceQueryRecord rRec(scene, sampler);
		Point2 apertureSample(0.5f);
		Float timeSample = 0.5f;
		RayDifferential sensorRay;

		block->clear();

		uint32_t queryType = RadianceQueryRecord::ESensorRay;

		if (!sensor->getFilm()->hasAlpha()) /* Don't compute an alpha channel if we don't have to */
			queryType &= ~RadianceQueryRecord::EOpacity;

		for (size_t i = 0; i<points.size(); ++i) {
			Point2i offset = Point2i(points[i]) + Vector2i(block->getOffset());
			if (stop)
				break;

			sampler->generate(offset);

			for (size_t j = 0; j<sampler->getSampleCount(); j++) {
				rRec.newQuery(queryType, sensor->getMedium());
				Point2 samplePos(Point2(offset) + Vector2(rRec.nextSample2D()));

				if (needsApertureSample)
					apertureSample = rRec.nextSample2D();
				if (needsTimeSample)
					timeSample = rRec.nextSample1D();

				Spectrum spec = sensor->sampleRayDifferential(
					sensorRay, samplePos, apertureSample, timeSample);

				sensorRay.scaleDifferential(diffScaleFactor);

				spec *= Li(sensorRay, rRec, offset);
				block->put(samplePos, spec, rRec.alpha);
				sampler->advance();
			}
		}
	}

	Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
		return Spectrum();
	}

	Point2 LocalDir2Coordinate(const Vector &wo) const {
		Point2 pp;
		float at = std::atan2(wo.y, wo.x) / (2 * M_PI);
		// float at = FastArcTan(wo.y / wo.x) / (2 * M_PI);
		pp.x = at >= 0 ? at : 1.0f + at;
		pp.y = (1.0f - wo.z * wo.z);
		pp.x = std::min(std::max(pp.x, 0.0f), 1.0f - 1e-6f);
		pp.y = std::min(std::max(pp.y, 0.0f), 1.0f - 1e-6f);
		return pp;
	};

	int BlockDivide(const Point2 &sample) const{
		const float stride = 1.0f / m_lightFieldResolution;
		int idx_x = std::min((int)(sample.x / stride), m_lightFieldResolution - 1);
		int idx_y = std::min((int)(sample.y / stride), m_lightFieldResolution - 1);
		int idx = idx_y * m_lightFieldResolution + idx_x;
		return idx;
	}

	Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec, const Point2i &offset) const {
		/* Some aliases and local variables */
		const Scene *scene = rRec.scene;
		Intersection &its = rRec.its;
		RayDifferential ray(r);
		Spectrum Li(0.0f);
		bool scattered = false;

		/* Perform the first ray intersection (or ignore if the
		   intersection has already been provided). */
		rRec.rayIntersect(ray);
		ray.mint = Epsilon;

		Spectrum throughput(1.0f);
		Float eta = 1.0f;

		bool foundRough = false;
		Spectrum roughAlbedo(1.f);
		Float roughDepth = 0.f;
		Spectrum LiDiffuse(0.f);
		Spectrum throughputDiffuse(1.f);
		Spectrum LiTemp;

		Vector firstHitDir;
		Vector firstHitNormal;
		Spectrum firstHitLiDelta(0.f);

		while (rRec.depth <= m_maxDepth || m_maxDepth < 0) {
			if (!its.isValid()) {
				/* If no intersection could be found, potentially return
				   radiance from a environment luminaire if it exists */
				if ((rRec.type & RadianceQueryRecord::EEmittedRadiance)
					&& (!m_hideEmitters || scattered)) {
					LiTemp = scene->evalEnvironment(ray);
					Li += LiTemp * throughput;
					if(foundRough)
						LiDiffuse += LiTemp * throughputDiffuse;
				}
				break;
			}

			const BSDF *bsdf = its.getBSDF(ray);

			/* Possibly include emitted radiance if requested */
			if (its.isEmitter() && (rRec.type & RadianceQueryRecord::EEmittedRadiance)
				&& (!m_hideEmitters || scattered)) {
				LiTemp = its.Le(-ray.d);
				Li += LiTemp * throughput;
				if(foundRough)
					LiDiffuse += LiTemp * throughputDiffuse;
			}

			/* Include radiance from a subsurface scattering model if requested */
			if (its.hasSubsurface() && (rRec.type & RadianceQueryRecord::ESubsurfaceRadiance)) {
				LiTemp = its.LoSub(scene, rRec.sampler, -ray.d, rRec.depth);
				Li += LiTemp * throughput;
				if(foundRough)
					LiDiffuse += LiTemp * throughputDiffuse;
			}

			if ((rRec.depth >= m_maxDepth && m_maxDepth > 0)
				|| (m_strictNormals && dot(ray.d, its.geoFrame.n)
				* Frame::cosTheta(its.wi) >= 0)) {

				/* Only continue if:
				   1. The current path length is below the specifed maximum
				   2. If 'strictNormals'=true, when the geometric and shading
				      normals classify the incident direction to the same side */
				break;
			}

			/* ==================================================================== */
			/*                     Direct illumination sampling                     */
			/* ==================================================================== */

			/* Estimate the direct illumination if this is requested */
			DirectSamplingRecord dRec(its);

			if (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance &&
				(bsdf->getType() & BSDF::ESmooth)) {
				Spectrum value = scene->sampleEmitterDirect(dRec, rRec.nextSample2D());
				if (!value.isZero()) {
					const Emitter *emitter = static_cast<const Emitter *>(dRec.object);

					/* Allocate a record for querying the BSDF */
					BSDFSamplingRecord bRec(its, its.toLocal(dRec.d), ERadiance);

					/* Evaluate BSDF * cos(theta) */
					const Spectrum bsdfVal = bsdf->eval(bRec);

					/* Prevent light leaks due to the use of shading normals */
					if (!bsdfVal.isZero() && (!m_strictNormals
						|| dot(its.geoFrame.n, dRec.d) * Frame::cosTheta(bRec.wo) > 0)) {

						/* Calculate prob. of having generated that direction
						   using BSDF sampling */
						Float bsdfPdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle)
							? bsdf->pdf(bRec) : 0;

						/* Weight using the power heuristic */
						Float weight = miWeight(dRec.pdf, bsdfPdf);
						LiTemp = value * weight;
						if (rRec.depth == 1)
							firstHitLiDelta += LiTemp * throughput * bsdfVal;
						Li += LiTemp * throughput * bsdfVal;
						if(!foundRough) {
							bRec.typeMask = BSDF::EDiffuse;
							const Spectrum bsdfValDiffuse = bsdf->eval(bRec);
							LiDiffuse += LiTemp * throughputDiffuse * bsdfValDiffuse;
						} else {
							LiDiffuse += LiTemp * throughputDiffuse * bsdfVal;
						}
					}
				}
			}

			/* ==================================================================== */
			/*                            BSDF sampling                             */
			/* ==================================================================== */

			/* Sample BSDF * cos(theta) */
			Float bsdfPdf;
			BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
			Spectrum bsdfWeight = bsdf->sample(bRec, bsdfPdf, rRec.nextSample2D());

			scattered |= bRec.sampledType != BSDF::ENull;

			if(scattered && !foundRough && bsdf->isRough(its)) {
				Vector rgb;
				roughAlbedo *= bsdf->getDiffuseReflectance(its);
				roughAlbedo.toLinearRGB(rgb.x, rgb.y, rgb.z);
				m_albedo->addSample(offset, rgb);
				Vector n = its.shFrame.n;
				if(Frame::cosTheta(its.wi) < 0) n = -n;
				m_normal->addSample(offset, n);
				firstHitNormal = n;
				m_depth->addSample(offset, Vector(roughDepth + its.t));

				BSDFSamplingRecord b(bRec);
				b.typeMask = BSDF::EDiffuse;
				Spectrum d = bsdf->eval(b, b.sampledType & BSDF::EDelta ? EDiscrete : ESolidAngle);
				throughputDiffuse *= d / bsdfPdf;

				firstHitDir = its.toWorld(bRec.wo);

				foundRough = true;
			} else {
				if(!foundRough) {
					if(scattered)
						roughAlbedo *= bsdf->getSpecularReflectance(its);
					roughDepth += its.t;
				}
				throughputDiffuse *= bsdfWeight;
			}

			if (bsdfWeight.isZero())
				break;

			/* Prevent light leaks due to the use of shading normals */
			const Vector wo = its.toWorld(bRec.wo);
			Float woDotGeoN = dot(its.geoFrame.n, wo);
			if (m_strictNormals && woDotGeoN * Frame::cosTheta(bRec.wo) <= 0)
				break;

			bool hitEmitter = false;
			Spectrum value;

			/* Trace a ray in this direction */
			ray = Ray(its.p, wo, ray.time);
			if (scene->rayIntersect(ray, its)) {
				/* Intersected something - check if it was a luminaire */
				if (its.isEmitter()) {
					value = its.Le(-ray.d);
					dRec.setQuery(ray, its);
					hitEmitter = true;
				}
			} else {
				/* Intersected nothing -- perhaps there is an environment map? */
				const Emitter *env = scene->getEnvironmentEmitter();

				if (env) {
					if (m_hideEmitters && !scattered)
						break;

					value = env->evalEnvironment(ray);
					if (!env->fillDirectSamplingRecord(dRec, ray))
						break;
					hitEmitter = true;
				} else {
					break;
				}
			}

			/* Keep track of the throughput and relative
			   refractive index along the path */
			throughput *= bsdfWeight;
			eta *= bRec.eta;

			/* If a luminaire was hit, estimate the local illumination and
			   weight using the power heuristic */
			if (hitEmitter &&
				(rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance)) {
				/* Compute the prob. of generating that direction using the
				   implemented direct illumination sampling technique */
				const Float lumPdf = (!(bRec.sampledType & BSDF::EDelta)) ?
					scene->pdfEmitterDirect(dRec) : 0;
				Spectrum LiTemp = value * miWeight(bsdfPdf, lumPdf);
				Li += LiTemp * throughput;
				if(foundRough)
					LiDiffuse += LiTemp * throughputDiffuse;
				if (rRec.depth == 1)
					firstHitLiDelta -= value * throughput * (1 - miWeight(bsdfPdf, lumPdf));
			}

			/* ==================================================================== */
			/*                         Indirect illumination                        */
			/* ==================================================================== */

			/* Set the recursive query type. Stop if no surface was hit by the
			   BSDF sample or if indirect illumination was not requested */
			if (!its.isValid() || !(rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance))
				break;
			rRec.type = RadianceQueryRecord::ERadianceNoEmission;

			if (rRec.depth++ >= m_rrDepth) {
				/* Russian roulette: try to keep path weights equal to one,
				   while accounting for the solid angle compression at refractive
				   index boundaries. Stop with at least some probability to avoid
				   getting stuck (e.g. due to total internal reflection) */

				Float q = std::min(throughput.max() * eta * eta, (Float) 0.95f);
				if (rRec.nextSample1D() >= q)
					break;
				throughput /= q;
				throughputDiffuse /= q;
			}
		}

		/* Store statistics */
		avgPathLength.incrementBase();
		avgPathLength += rRec.depth;

		if(!foundRough) {
			m_albedo->addSample(offset, Vector(0.f));
			m_normal->addSample(offset, -ray.d);
			m_depth->addSample(offset, Vector(0.f));
		}
		Vector v;
		LiDiffuse.toLinearRGB(v.x, v.y, v.z);
		m_diffuse->addSample(offset, v);
		Spectrum LiSpecular = Li - LiDiffuse;
		LiSpecular.toLinearRGB(v.x, v.y, v.z);
		m_specular->addSample(offset, v);

		int lightFieldBlockIdx = BlockDivide(LocalDir2Coordinate(Frame(firstHitNormal).toLocal(firstHitDir)));
		Spectrum firstHitLi = (Li - firstHitLiDelta) / roughAlbedo;
		if(firstHitLi.isValid()) {
			Vector rgb;
			firstHitLi.toLinearRGB(rgb.x, rgb.y, rgb.z);
			m_lightFieldBlocks[lightFieldBlockIdx]->addSample(offset, rgb);
		}

		return Li;
	}

	inline Float miWeight(Float pdfA, Float pdfB) const {
		pdfA *= pdfA;
		pdfB *= pdfB;
		return pdfA / (pdfA + pdfB);
	}

	void serialize(Stream *stream, InstanceManager *manager) const {
		MonteCarloIntegrator::serialize(stream, manager);
	}

	std::string toString() const {
		std::ostringstream oss;
		oss << "MIPathTracer[" << endl
			<< "  maxDepth = " << m_maxDepth << "," << endl
			<< "  rrDepth = " << m_rrDepth << "," << endl
			<< "  strictNormals = " << m_strictNormals << endl
			<< "]";
		return oss.str();
	}

	MTS_DECLARE_CLASS()

private:
	mutable std::unique_ptr<OutputBuffer> m_albedo;
	mutable std::unique_ptr<OutputBuffer> m_normal;
	mutable std::unique_ptr<OutputBuffer> m_depth;
	mutable std::unique_ptr<OutputBuffer> m_diffuse;
	mutable std::unique_ptr<OutputBuffer> m_specular;

	int m_lightFieldResolution;
	int m_lightFieldBlockCount;
	mutable std::vector<std::unique_ptr<OutputBuffer>> m_lightFieldBlocks;
};

MTS_IMPLEMENT_CLASS_S(MIPathTracer, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(MIPathTracer, "MI path tracer");
MTS_NAMESPACE_END
