import Foundation

public enum Llama {
  enum LlamaError: Error {
    case insufficientData
  }

  static func loadValue<T: FixedWidthInteger>(
    from data: Data, offset: inout Int
  ) throws -> T {
    let byteSize = MemoryLayout<T>.stride
    let range = offset..<(offset + byteSize)
    guard range.upperBound <= data.count else {
      throw LlamaError.insufficientData
    }
    let value = data.withUnsafeBytes { rawBufferPointer in
      rawBufferPointer.loadUnaligned(fromByteOffset: offset, as: T.self)
    }.littleEndian
    offset += byteSize
    return value
  }

  struct Config {
    /// transformer dimension
    let dimension: Int
    /// for ffn layers
    let hiddenDimension: Int
    /// number of layers
    let layerCount: Int
    /// number of query heads
    let headCount: Int
    /// number of key/value heads
    let keyValueHeadCount: Int
    /// vocabulary size
    let rawVocabularySize: Int
    /// vocabulary size(absolute)
    let vocabularySize: Int
    /// max ksequence length
    let sequenceLength: Int

    init(from data: Data, offset: inout Int) throws {
      self.dimension = Int(try loadValue(from: data, offset: &offset) as Int32)
      self.hiddenDimension = Int(try loadValue(from: data, offset: &offset) as Int32)
      self.layerCount = Int(try loadValue(from: data, offset: &offset) as Int32)
      self.headCount = Int(try loadValue(from: data, offset: &offset) as Int32)
      self.keyValueHeadCount = Int(try loadValue(from: data, offset: &offset) as Int32)
      self.rawVocabularySize = Int(try loadValue(from: data, offset: &offset) as Int32)
      self.vocabularySize = abs(rawVocabularySize)
      self.sequenceLength = Int(try loadValue(from: data, offset: &offset) as Int32)
    }

    public func toString() -> String {
      return """
        dimension: \(dimension)
        hiddenDimension: \(hiddenDimension)
        layerCount: \(layerCount)
        headCount: \(headCount)
        keyValueHeadCount: \(keyValueHeadCount)
        vocabularySize: \(vocabularySize)
        sequenceLength: \(sequenceLength)
        """
    }
  }

  struct TransformerWeights {
    private let modelData: Data
    let config: Config

    /// token embedding table
    let tokenEmbeddingTable: FloatBufferView
    /// rmsnorm weights(attention)
    let attentionRMSNormWeights: [FloatBufferView]
    /// rmsnorm weights(ffn)
    let feedForwardRMSNormWeights: [FloatBufferView]
    /// weights for matmuls(query)
    let attentionQueryWeights: [FloatBufferView]
    /// weights for matmuls(key)
    let attentionKeyWeights: [FloatBufferView]
    /// weights for matmuls(value)
    let attentionValueWeights: [FloatBufferView]
    /// weights for matmuls(output)
    let attentionOutputWeights: [FloatBufferView]
    /// weights for ffn(1st layer)
    let feedForwardLayer1Weights: [FloatBufferView]
    /// weights for ffn(2nd layer)
    let feedForwardLayer2Weights: [FloatBufferView]
    /// weights for ffn(3rd layer)
    let feedForwardLayer3Weights: [FloatBufferView]
    /// final rmsnorm weights
    let finalRMSNormWeights: FloatBufferView
    /// classifier weights(output logits)
    let outputClassifierWeights: FloatBufferView

    init(config: Config, modelData: Data, sharedWeights: Bool, currentOffset: inout Int) throws {
      self.config = config
      self.modelData = modelData
      let floatStride = MemoryLayout<Float>.stride

      let headSize = config.dimension / config.headCount

      func createView(count: Int) -> FloatBufferView {
        let byteSize = count * floatStride
        let range = currentOffset..<(currentOffset + byteSize)
        guard range.upperBound <= modelData.count else {
          fatalError("Invalid range: \(range) exceeds modelData size")
        }
        let view = FloatBufferView(data: modelData, range: range)
        currentOffset += byteSize
        return view
      }

      func createViews(layers: Int, countPerLayer: Int) -> [FloatBufferView] {
        var views = [FloatBufferView]()
        views.reserveCapacity(layers)
        for _ in 0..<layers {
          views.append(
            createView(count: countPerLayer)
          )
        }
        return views
      }

      // (vocabularySize, dimension)
      self.tokenEmbeddingTable = createView(count: config.vocabularySize * config.dimension)
      // (layerCount, dimension) rmsnorm weights
      self.attentionRMSNormWeights = createViews(
        layers: config.layerCount,
        countPerLayer: config.dimension
      )
      // (layerCount, dimension, headCount * headSize)
      self.attentionQueryWeights = createViews(
        layers: config.layerCount,
        countPerLayer: config.dimension * config.dimension
      )
      // (layerCount, dimension, keyValueHeadCount * headSize)
      self.attentionKeyWeights = createViews(
        layers: config.layerCount,
        countPerLayer: config.dimension * (config.keyValueHeadCount * headSize)
      )
      // (layerCount, dimension, keyValueHeadCount * headSize)
      self.attentionValueWeights = createViews(
        layers: config.layerCount,
        countPerLayer: config.dimension * (config.keyValueHeadCount * headSize)
      )
      // (layerCount, headCount * headSize, dimension)
      self.attentionOutputWeights = createViews(
        layers: config.layerCount,
        countPerLayer: config.dimension * config.dimension
      )
      // (layerCount, dimension)
      self.feedForwardRMSNormWeights = createViews(
        layers: config.layerCount,
        countPerLayer: config.dimension
      )
      // (layerCount, hiddenDimension, dimension)
      self.feedForwardLayer1Weights = createViews(
        layers: config.layerCount,
        countPerLayer: config.dimension * config.hiddenDimension
      )
      // (layerCount, dimension, hiddenDimension)
      self.feedForwardLayer2Weights = createViews(
        layers: config.layerCount,
        countPerLayer: config.hiddenDimension * config.dimension
      )
      // (layerCount, hiddenDimension, dimension)
      self.feedForwardLayer3Weights = createViews(
        layers: config.layerCount,
        countPerLayer: config.dimension * config.hiddenDimension
      )
      // (dimension, )
      self.finalRMSNormWeights = createView(count: config.dimension)

      // NOTE: Skip freq_cis_real and freq_cis_imag
      let freqCisSize = config.sequenceLength * headSize / 2
      currentOffset += freqCisSize
      currentOffset += freqCisSize

      // check range
      guard currentOffset <= modelData.count else {
        fatalError("Invalid range: \(currentOffset) exceeds modelData size")
      }

      // (vocabularySize, dimension)
      if sharedWeights {
        let range = 0..<(config.vocabularySize * config.dimension * floatStride)
        self.outputClassifierWeights = FloatBufferView(data: modelData, range: range)
      } else {
        self.outputClassifierWeights = createView(count: config.vocabularySize * config.dimension)
      }

      if currentOffset != modelData.count {
        print("Warning: \(currentOffset) does not match modelData size \(modelData.count)")
      }
      print("Successfully loaded weights. Final offset: \(currentOffset)")
    }
  }

  struct RunState {
    let activationBuffer: [Float]
    let activationBuffer2: [Float]
    let ffnHiddenBuffer: [Float]
    let ffnHiddenBuffer2: [Float]
    let query: [Float]
    let attributionScores: [[Float]]
    let logits: [Float]
    let keyCashe: [[[Float]]]
    let valueCashe: [[[Float]]]

    init(config: Config) {
      let keyValueDimension = (config.dimension * config.keyValueHeadCount) / config.headCount
      func makeFloatArray(_ dim1: Int, _ dim2: Int) -> [[Float]] {
        return [[Float]](repeating: [Float](repeating: 0, count: dim2), count: dim1)
      }
      func makeFloatArray(_ dim1: Int, _ dim2: Int, _ dim3: Int) -> [[[Float]]] {
        return [[[Float]]](
          repeating: makeFloatArray(dim1, dim2),
          count: dim3)
      }

      self.activationBuffer = [Float](repeating: 0, count: config.dimension)
      self.activationBuffer2 = [Float](repeating: 0, count: config.dimension)
      self.ffnHiddenBuffer = [Float](repeating: 0, count: config.hiddenDimension)
      self.ffnHiddenBuffer2 = [Float](repeating: 0, count: config.hiddenDimension)
      self.query = [Float](repeating: 0, count: config.dimension)
      self.keyCashe = makeFloatArray(config.layerCount, config.sequenceLength, keyValueDimension)
      self.valueCashe = makeFloatArray(config.layerCount, config.sequenceLength, keyValueDimension)
      self.attributionScores = makeFloatArray(config.sequenceLength, config.headCount)
      self.logits = [Float](repeating: 0, count: config.vocabularySize)
    }
  }

  struct Transformer {
    let config: Config
    let weights: TransformerWeights
    var runState: RunState

    private let modelWeightsData: Data

    init(checkpointURL: URL) throws {
      self.modelWeightsData = try Data(contentsOf: checkpointURL, options: .mappedIfSafe)
      var currentOffset = 0
      self.config = try Config(from: modelWeightsData, offset: &currentOffset)
      self.weights = try TransformerWeights(
        config: config,
        modelData: modelWeightsData,
        sharedWeights: config.rawVocabularySize > 0,
        currentOffset: &currentOffset
      )
      self.runState = RunState(config: config)
    }
  }

  static func rmsnorm(
    out: inout [Float], activation: [Float], weight: FloatBufferView, size: Int
  ) {
    var sumOfSquares: Float = 0
    for i in 0..<size {
      let value = activation[i]
      sumOfSquares += value * value
    }
    sumOfSquares = sumOfSquares / Float(size)
    sumOfSquares = sumOfSquares + 1e-5
    sumOfSquares = 1 / sqrt(sumOfSquares)
    for i in 0..<size {
      out[i] = weight[i] * (sumOfSquares * activation[i])
    }
  }
}
