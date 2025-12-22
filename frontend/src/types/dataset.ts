export type DatasetStatus = 'uploaded' | 'processing' | 'ready' | 'error';

export interface Dataset {
  id: string;
  name: string;
  filename: string;
  size: number;
  rowCount: number;
  columnCount: number;
  createdAt: string;
  updatedAt: string;
  status: DatasetStatus;
}

export interface ColumnInfo {
  name: string;
  dataType: string;
  nullCount: number;
  uniqueCount: number;
  sampleValues: any[];
}

export interface DatasetStats {
  rowCount: number;
  columnCount: number;
  numericColumns: number;
  categoricalColumns: number;
  missingValues: number;
  duplicateRows: number;
  memoryUsage: number;
}

export interface VisualizationData {
  type: 'histogram' | 'scatter' | 'box' | 'correlation' | 'missing';
  data: any;
  title: string;
  xLabel?: string;
  yLabel?: string;
}
