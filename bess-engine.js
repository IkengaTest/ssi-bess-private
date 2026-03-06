/**
 * BESS Analytics Engine
 * Client-side analytics for SSI-ENN BESS Valuation
 */

class BESSEngine {
  constructor() {
    this.data = null;
    this.substations = [];
    this.regions = [];
  }

  async loadData(url = 'data.json') {
    try {
      const response = await fetch(url);
      this.data = await response.json();
      this.substations = this.data.substations || [];
      this.regions = this.extractRegions();
      return true;
    } catch (error) {
      console.error('Error loading data:', error);
      return false;
    }
  }

  extractRegions() {
    const regionMap = new Map();
    this.substations.forEach(sub => {
      if (!regionMap.has(sub.region)) {
        regionMap.set(sub.region, []);
      }
      regionMap.get(sub.region).push(sub);
    });
    return Array.from(regionMap.entries()).map(([name, subs]) => ({
      name,
      substations: subs
    }));
  }

  // Filter operations
  filterByRegion(regionName) {
    return this.substations.filter(s => s.region === regionName);
  }

  filterByBand(bandName) {
    const bandLower = bandName.toLowerCase();
    return this.substations.filter(s => s.classification && s.classification.toLowerCase() === bandLower);
  }

  filterByVoltage(minVoltage, maxVoltage) {
    return this.substations.filter(s => {
      const v = s.voltage_kv || 0;
      return v >= minVoltage && v <= maxVoltage;
    });
  }

  filterByIRR(minIRR) {
    return this.substations.filter(s => {
      const irr = s.bess?.config_B?.IRR_pct || 0;
      return irr >= minIRR;
    });
  }

  filterByNPV(minNPV) {
    return this.substations.filter(s => {
      const npv = s.bess?.config_B?.NPV_M || 0;
      return npv >= minNPV;
    });
  }

  // Sorting operations
  sortByField(data, field, ascending = true) {
    const sorted = [...data];
    sorted.sort((a, b) => {
      const aVal = this.getNestedValue(a, field);
      const bVal = this.getNestedValue(b, field);
      if (aVal === null || aVal === undefined) return 1;
      if (bVal === null || bVal === undefined) return -1;
      return ascending ? (aVal > bVal ? 1 : -1) : (aVal < bVal ? 1 : -1);
    });
    return sorted;
  }

  getNestedValue(obj, path) {
    return path.split('.').reduce((acc, part) => acc?.[part], obj);
  }

  // Aggregation operations
  computeStats(data, field) {
    const values = data
      .map(item => this.getNestedValue(item, field))
      .filter(v => v !== null && v !== undefined && !isNaN(v));
    
    if (values.length === 0) return null;

    const sorted = values.sort((a, b) => a - b);
    const sum = values.reduce((a, b) => a + b, 0);
    const mean = sum / values.length;
    const median = sorted.length % 2 === 0
      ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
      : sorted[Math.floor(sorted.length / 2)];

    return {
      min: sorted[0],
      max: sorted[sorted.length - 1],
      mean,
      median,
      sum,
      count: values.length
    };
  }

  // Band statistics
  getBandDistribution(data = this.substations) {
    const distribution = {
      Low: 0,
      Medium: 0,
      High: 0,
      Critical: 0
    };

    data.forEach(sub => {
      const band = sub.classification || 'Unknown';
      if (distribution.hasOwnProperty(band)) {
        distribution[band]++;
      }
    });

    return distribution;
  }

  // KPI computations
  computeKPIs(data = this.substations) {
    const bandDist = this.getBandDistribution(data);
    const rStats = this.computeStats(data, 'R_median');
    const npvBStats = this.computeStats(data, 'bess.config_B.NPV_M');
    const irrStats = this.computeStats(data, 'bess.config_B.IRR_pct');
    const npvAStats = this.computeStats(data, 'bess.config_A.NPV_M');

    const provinces = new Set(data.map(s => s.province));
    const topProvince = this.getTopProvinceByNPV(data);

    return {
      totalSubstations: data.length,
      fleetMedianR: rStats?.median || 0,
      totalInvestmentNPV: npvBStats?.sum || 0, // already in millions
      medianIRR: irrStats?.median || 0,
      configAMedianNPV: npvAStats?.median || 0,
      configBMedianNPV: npvBStats?.median || 0,
      highCriticalCount: (bandDist.High || 0) + (bandDist.Critical || 0),
      topProvince: topProvince,
      bandDistribution: bandDist,
      totalProvinces: provinces.size
    };
  }

  // Get top province
  getTopProvinceByNPV(data = this.substations) {
    const provinceMap = new Map();
    data.forEach(sub => {
      const prov = sub.province || 'Unknown';
      if (!provinceMap.has(prov)) {
        provinceMap.set(prov, 0);
      }
      provinceMap.set(prov, provinceMap.get(prov) + (sub.bess?.config_B?.NPV_M || 0));
    });

    let topProv = 'N/A';
    let maxNPV = 0;
    provinceMap.forEach((npv, prov) => {
      if (npv > maxNPV) {
        maxNPV = npv;
        topProv = prov;
      }
    });
    return topProv;
  }

  // Get top opportunities
  getTopOpportunities(data = this.substations, limit = 20) {
    return this.sortByField(data, 'bess.config_B.NPV_M', false).slice(0, limit);
  }

  // Get region summary
  getRegionSummary(regionName) {
    const subs = this.filterByRegion(regionName);
    if (subs.length === 0) {
      return null;
    }

    const bandDist = this.getBandDistribution(subs);

    return {
      region: regionName,
      count: subs.length,
      medianR: this.computeStats(subs, 'R_median')?.median || 0,
      bandDistribution: bandDist,
      medianNPVConfigA: this.computeStats(subs, 'bess.config_A.NPV_M')?.median || 0,
      medianNPVConfigB: this.computeStats(subs, 'bess.config_B.NPV_M')?.median || 0,
      medianIRR: this.computeStats(subs, 'bess.config_B.IRR_pct')?.median || 0,
      topProvince: this.getTopProvinceByNPV(subs)
    };
  }

  // Get region top substations
  getRegionTopSubstations(regionName, limit = 5) {
    const subs = this.filterByRegion(regionName);
    return this.sortByField(subs, 'bess.config_B.NPV_M', false).slice(0, limit);
  }

  // Search
  search(query, fields = ['name', 'province', 'region']) {
    const q = query.toLowerCase();
    return this.substations.filter(sub =>
      fields.some(field => {
        const val = this.getNestedValue(sub, field);
        return val && val.toString().toLowerCase().includes(q);
      })
    );
  }

  // Get substation by ID or name
  getSubstation(identifier) {
    return this.substations.find(s =>
      s.id === identifier || s.name === identifier
    );
  }

  // Geography helpers
  getAllProvinces() {
    return [...new Set(this.substations.map(s => s.province))].sort();
  }

  getAllRegions() {
    return this.regions.map(r => r.name).sort();
  }

  getBands() {
    return ['Low', 'Medium', 'High', 'Critical'];
  }

  getVoltageRange() {
    const stats = this.computeStats(this.substations, 'voltage_kv');
    return stats ? { min: stats.min, max: stats.max } : { min: 0, max: 400 };
  }

  // Format utilities
  formatNumber(num, decimals = 2) {
    if (num === null || num === undefined) return 'N/A';
    return Number(num).toLocaleString('it-IT', { maximumFractionDigits: decimals });
  }

  formatCurrency(num, decimals = 0) {
    if (num === null || num === undefined) return 'N/A';
    return '€' + Number(num).toLocaleString('it-IT', { maximumFractionDigits: decimals });
  }

  formatPercent(num, decimals = 1) {
    if (num === null || num === undefined) return 'N/A';
    return Number(num).toFixed(decimals) + '%';
  }

  formatYears(num, decimals = 1) {
    if (num === null || num === undefined) return 'N/A';
    return Number(num).toFixed(decimals) + ' y';
  }

  // Band color helpers
  getBandColor(band) {
    const colors = {
      'Low': '#5d8563',
      'Medium': '#b8863a',
      'High': '#aa4234',
      'Critical': '#941914'
    };
    return colors[band] || '#8a7e76';
  }

  getBandBgColor(band) {
    const colors = {
      'Low': 'rgba(93,133,99,0.10)',
      'Medium': 'rgba(184,134,58,0.10)',
      'High': 'rgba(170,66,52,0.10)',
      'Critical': 'rgba(148,25,20,0.10)'
    };
    return colors[band] || 'rgba(138,126,118,0.10)';
  }
}

// Global engine instance
const engine = new BESSEngine();
