import Accelerate
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
        self.outputClassifierWeights = self.tokenEmbeddingTable
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
    var activationBuffer: [Float]
    var activationBuffer2: [Float]
    var ffnHiddenBuffer: [Float]
    var ffnHiddenBuffer2: [Float]
    var query: [Float]
    var key: [Float]
    var value: [Float]
    var attributionScores: [[Float]]
    var logits: [Float]
    var keyCache: [[[Float]]]
    var valueCache: [[[Float]]]

    init(config: Config) {
      let keyValueDimension = (config.dimension * config.keyValueHeadCount) / config.headCount

      // n次元配列を作成するためのヘルパー関数
      func makeFloatArray(_ dim1: Int, _ dim2: Int) -> [[Float]] {
        return [[Float]](repeating: [Float](repeating: 0, count: dim2), count: dim1)
      }
      func makeFloatArray(_ dim1: Int, _ dim2: Int, _ dim3: Int) -> [[[Float]]] {
        return [[[Float]]](
          repeating: makeFloatArray(dim2, dim3),
          count: dim1)
      }

      self.activationBuffer = [Float](repeating: 0, count: config.dimension)
      self.activationBuffer2 = [Float](repeating: 0, count: config.dimension)
      self.ffnHiddenBuffer = [Float](repeating: 0, count: config.hiddenDimension)
      self.ffnHiddenBuffer2 = [Float](repeating: 0, count: config.hiddenDimension)
      self.query = [Float](repeating: 0, count: config.dimension)
      self.key = [Float](repeating: 0, count: keyValueDimension)
      self.value = [Float](repeating: 0, count: keyValueDimension)
      // float[layerCount][sequenceLength][keyValueDimension]
      self.keyCache = makeFloatArray(config.layerCount, config.sequenceLength, keyValueDimension)
      self.valueCache = makeFloatArray(config.layerCount, config.sequenceLength, keyValueDimension)
      // float[headCount][sequenceLength]
      self.attributionScores = makeFloatArray(config.headCount, config.sequenceLength)
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

    mutating func forward(token: Int, position: Int) {
      let dimension = config.dimension
      let keyValueDimension = (config.dimension * config.keyValueHeadCount) / config.headCount
      let keyValueMultiplier = config.headCount / config.keyValueHeadCount
      let hiddenDimension = config.hiddenDimension
      let headSize = dimension / config.headCount
      var headAsqrt: Float = sqrt(Float(headSize))
      let activation = weights.tokenEmbeddingTable.slice(from: token, size: dimension)
      var currentActivation: [Float] = activation.toArray()

      // forward all the layers
      for layer in 0..<config.layerCount {
        // attention rmsnorm
        rmsnorm(
          out: &runState.activationBuffer, activation: currentActivation,
          weight: weights.attentionRMSNormWeights[layer], size: dimension)

        // key and value point to the kv cache
        runState.key = runState.keyCache[layer][position]
        runState.value = runState.valueCache[layer][position]

        // query key value matmuls for this position
        matmul(
          activationOut: &runState.query, activation: runState.activationBuffer,
          weight: weights.attentionQueryWeights[layer], n: dimension, d: dimension)
        matmul(
          activationOut: &runState.key, activation: runState.activationBuffer,
          weight: weights.attentionKeyWeights[layer], n: keyValueDimension, d: dimension)
        matmul(
          activationOut: &runState.value, activation: runState.activationBuffer,
          weight: weights.attentionValueWeights[layer], n: keyValueDimension, d: dimension)

        // RoPE relative positional encoding: complex-valued rotate query and key in each head
        for i in stride(from: 0, to: dimension, by: 2) {
          let headDimension = i % headSize
          let frequency = 1.0 / pow(10000.0, Float(headDimension) / Float(headSize))
          let value = Float(position) * frequency
          let rotationCosine = cos(value)
          let rotationSine = sin(value)
          let rotationTargetCount = i < keyValueDimension ? 2 : 1
          for v in 0..<rotationTargetCount {
            let targetVector = v == 0 ? runState.query : runState.key
            let v0 = targetVector[i]
            let v1 = targetVector[i + 1]
            let res0 = v0 * rotationCosine - v1 * rotationSine
            let res1 = v0 * rotationSine + v1 * rotationCosine
            if v == 0 {
              runState.query[i] = res0
              runState.query[i + 1] = res1
            } else {
              runState.key[i] = res0
              runState.key[i + 1] = res1
            }
          }
        }

      }
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
    sumOfSquares /= Float(size)
    sumOfSquares += 1e-5
    sumOfSquares = 1 / sqrt(sumOfSquares)
    for i in 0..<size {
      out[i] = weight[i] * (sumOfSquares * activation[i])
    }
  }

  static func softmax(
    activation: inout [Float], size: Int
  ) {
    let maxValue = activation.max() ?? 0
    var sum: Float = 0
    for i in 0..<size {
      activation[i] = exp(activation[i] - maxValue)
      sum += activation[i]
    }
    for i in 0..<size {
      activation[i] /= sum
    }
  }

  static func matmul(
    activationOut: inout [Float],
    activation: [Float],
    weight: FloatBufferView,
    n: Int,  // 列数
    d: Int,  // 行数
  ) {
    precondition(activationOut.count == d, "activationOut count must be \(d)")
    precondition(activation.count == n, "activation count must be \(n)")
    precondition(weight.count == n * d, "weight count must be \(n * d)")

    if d == 0 || n == 0 {
      for i in 0..<activationOut.count {
        activationOut[i] = 0
      }
      return
    }

    weight.withUnsafeBufferPointer { weightBufferPtr in
      guard let weightMatrixPtr = weightBufferPtr.baseAddress else {
        if d > 0 && n > 0 {  // d,nが0でないのにポインタがnilなら問題
          print("Error: Weight buffer pointer is nil for non-empty matrix dimensions.")
          for i in 0..<activationOut.count { activationOut[i] = 0.0 }
        }
        return
      }
      cblas_sgemv(
        CblasRowMajor,  // Order：行優先で保存されている ex. weight[i * n + j]
        CblasNoTrans,  // TransA：転置しない
        CLong(d),  // M：行数
        CLong(n),  // N：列数
        1.0,  // alpha：係数
        weightMatrixPtr,  // A：重み行列データのポインタ
        CLong(n),  // LeadingDimensionA：行優先で転置しない場合行列の列数と同じ
        activation,  // X：入力ベクトル
        CLong(1),  // strideX：要素間のストライド
        0.0,  // beta：既存の値を加算する際にかける係数
        &activationOut,  // Y：出力ベクトル
        CLong(1)  // strideY：要素間のストライド
      )
    }
  }
}
